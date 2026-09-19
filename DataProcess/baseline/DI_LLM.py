import argparse
import math
import os
import random
import re
import string
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from openai import OpenAI
import pandas as pd
from sympy import factorint
from tqdm.auto import tqdm
from tree_sitter import Language, Parser

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

lang = Language(os.environ.get("TS_LANG_LIB", "build/my-languages.so"), "cpp")

def get_node_text_bytes(node, src_bytes):
    return src_bytes[node.start_byte:node.end_byte]

def iter_nodes(node):
    """递归迭代节点（先序）"""
    yield node
    for ch in node.children:
        yield from iter_nodes(ch)

def apply_replacements_bytes(src_bytes, replacements):
    """
    replacements: list of (start_byte, end_byte, replacement_bytes)
    """
    if not replacements:
        return src_bytes
    replacements_sorted = sorted(replacements, key=lambda x: x[0], reverse=True)
    out = bytearray(src_bytes)
    for start, end, new_bytes in replacements_sorted:
        if start < 0 or end < start or end > len(out):
            continue
        out[start:end] = new_bytes
    return bytes(out)

c_keywords = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__", "uint8_t", "uint16_t",
                "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t",
                "int64_t", "bool"]
c_macros = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",
              "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
              "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]
c_special_ids = ["main","std",
                   "stdio", "cstdio", "stdio.h",
                   "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",
                   "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush",
                   "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                   "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                   "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                   "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                   "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                   "rewind", "clearerr", "feof", "ferror", "perror", "getline",
                   "stdlib", "cstdlib", "stdlib.h",
                   "size_t", "div_t", "ldiv_t", "lldiv_t",
                   "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",
                   "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                   "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                   "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                   "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                   "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                   "mbstowcs", "wcstombs",
                   "string", "cstring", "string.h",
                   "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",
                   "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                   "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                   "strpbrk" ,"strstr", "strtok", "strxfrm",
                   "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",
                   "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                   "iostream", "istream", "ostream", "fstream", "sstream",
                   "iomanip", "iosfwd",
                   "ios", "wios", "streamoff", "streampos", "wstreampos",
                   "streamsize", "cout", "cerr", "clog", "cin",
                   "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",
                   "noshowbase", "showpoint", "noshowpoint", "showpos",
                   "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                   "left", "right", "internal", "dec", "oct", "hex", "fixed",
                   "scientific", "hexfloat", "defaultfloat", "width", "fill",
                   "precision", "endl", "ends", "flush", "ws", "showpoint",
                   "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",
                   "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                   "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                   "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]

_identifier_token_type = "identifier"
_function_nodes = {
    "function_definition",
    "function_declarator",
    "method_definition",
    "constructor_declarator",
    "method_declaration",
    "call_expression"
}
_variable_nodes = {
    "init_declarator",
    "declarator",
    "variable_declarator",
    "field_declaration",
    "variable_declaration",
    "local_variable_declaration",
    "parameter_declaration",
    "parameter",
    "formal_parameter",
    "member_expression",
    "scoped_identifier",
    "argument_list",
    "member_expression",
    "attribute"
}

# ---------- Dead-code insertion ----------
_statement_parent_types = {
    "compound_statement"
}

def insert_dead_code(code, snippet_template="int {name} = {value};", seed=None):
    if seed is not None:
        random.seed(seed)

    parser = Parser()
    parser.set_language(lang)
    src_bytes = code.encode("utf8")
    tree = parser.parse(src_bytes)
    root = tree.root_node

    candidate_positions = []
    for node in iter_nodes(root):
        if node.type == "compound_statement":
            for child in node.children:
                if child.is_named:
                    candidate_positions.append(child.end_byte)

    if candidate_positions:
        insert_pos = random.choice(candidate_positions)
    else:
        insert_pos = len(src_bytes)

    existing_names = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', code))
    base = "ret_var_"
    idx = random.randint(1000, 9999)
    name = f"{base}{idx}"
    while name in existing_names:
        idx = random.randint(1000, 9999)
        name = f"{base}{idx}"
    value = random.randint(1000, 9999)
    snippet = snippet_template.format(name=name, value=value)
    prev_nl = src_bytes.rfind(b"\n", 0, insert_pos)
    line_start = prev_nl + 1 if prev_nl >= 0 else 0
    i = line_start
    while i < insert_pos and src_bytes[i] in (b" "[0], b"\t"[0]):
        i += 1
    indent_bytes = src_bytes[line_start:i]
    insertion_bytes = b"\n" + indent_bytes + snippet.encode("utf8")
    new_bytes = apply_replacements_bytes(src_bytes, [(insert_pos, insert_pos, insertion_bytes)])
    return new_bytes.decode("utf8")

def find_descendants_by_type(root, type_name):
    res = []
    for node in iter_nodes(root):
        if node.type == type_name:
            res.append(node)
    return res

def find_function_def_or_first_compound(root):
    funcs = find_descendants_by_type(root, "function_definition")
    if funcs:
        return funcs[0]
    compounds = find_descendants_by_type(root, "compound_statement")
    if compounds:
        return compounds[0]
    return None

def extract_statements_in_compound(comp_node, source_bytes):
    stmts = []
    for n in comp_node.named_children:
        stmts.append((n, n.start_byte, n.end_byte, get_node_text_bytes(n, source_bytes).decode('utf8', errors='ignore')))
    return stmts

def extract_code_from_text(text):
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            s = parts[i]
            s = re.sub(r'^\s*[a-zA-Z0-9\+\-]+\s*\n', '', s)
            if s.strip():
                return s.strip()
    return text.strip()

def generate_candidates_openai(context_prefix, n=5, temperature=0.8, model=None):
    client = OpenAI(api_key='')

    prompt = (
        "You are a C++ code generator. Given the prefix of a function (the code before the insertion point), "
        "generate a short C++ snippet (3-5 lines) to insert immediately after the cursor. Instructions:\n"
        "- The snippet MAY declare new variables and MAY call functions.\n"
        "- The snippet SHOULD NOT depend on variables from the surrounding function (i.e., do not use the parameters or local variables of the function). "
        "This will be automatically checked; if you cannot produce such snippet, output an empty response.\n"
        "- Do NOT include #include or using directives.\n"
        "- Output only the code snippet (no explanations).\n\n"
        "Function prefix (the code before insertion point):\n```c++\n" + context_prefix + "\n```\n\nSnippet:"
    )
    messages = [{"role": "user", "content": prompt}]
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, n=n)
    outputs = []
    for choice in resp.choices:
        txt = choice.message.content
        code = extract_code_from_text(txt)
        outputs.append(code)
    return outputs

def extract_var(root, src_bytes):
    decls = set()
    for node in iter_nodes(root):
        if node.type == _identifier_token_type:
            name = get_node_text_bytes(node, src_bytes).decode("utf8", errors="ignore")
            if name in c_keywords or name in c_macros or name in c_special_ids:
                continue
            decls.add(name)
    return decls

_basic_blacklist = [
        "system", "fork", "exec", "execl", "execle", "execlp", "execvp", "execv", "execve",
        "popen", "pclose", "dlopen", "dlsym", "dlclose",
        "malloc", "calloc", "realloc", "free", "new", "delete", "mmap", "munmap",
        "pthread_create", "pthread_join", "pthread_mutex_lock", "pthread_mutex_unlock",
        "exit", "abort", "kill", "raise", "longjmp", "setjmp", "throw", "asm", "return",
        "chmod", "chown", "remove", "rename", "mkdir", "rmdir"
    ]

def ast_blacklist_check(root, source_bytes):
    for node in iter_nodes(root):
        if node.type == "call_expression":
            callee = None
            for desc in iter_nodes(node):
                if desc.type in ("identifier", "scoped_identifier"):
                    callee = get_node_text_bytes(desc, source_bytes).decode("utf8", errors="ignore")
                    break
            if callee:
                base = callee.split("::")[-1]
                if base in _basic_blacklist:
                    return False

        if node.type in ("new_expression", "delete_expression", "throw_statement", "return_statement"):
            return False

        if node.type in ("scoped_identifier", "identifier"):
            name = get_node_text_bytes(node, source_bytes).decode("utf8", errors="ignore")
            if name in _basic_blacklist:
                return False

    return True

def validate_candidate_allow_calls(snippet, source_bytes, sel_end, parser, original_idents):
    new_source_bytes = source_bytes[:sel_end] + b"\n" + snippet.encode("utf8") + source_bytes[sel_end:]
    try:
        tree = parser.parse(new_source_bytes)
    except Exception as e:
        return False, f"tree-sitter parse failed: {e}"
    for node in iter_nodes(tree.root_node):
        if node.type == "ERROR":
            return False, f"tree-sitter parse ERROR at bytes ({node.start_byte},{node.end_byte})"

    try:
        p2 = Parser()
        p2.set_language(lang)
        snippet_bytes = snippet.encode("utf8")
        tree_snip = p2.parse(snippet_bytes)
    except Exception as e:
        return False, f"tree-sitter parse failed for snippet: {e}"

    if not ast_blacklist_check(tree_snip.root_node, snippet_bytes):
        return False, f"forbidden keyword."

    inter = extract_var(tree_snip.root_node, snippet_bytes) & original_idents if original_idents else set()
    if inter:
        return False, f"data dependency: snippet uses original function id(s): {sorted(list(inter))}"
    return True, "ok"

def lm_base_poison(code, n_candidates=5, seed=None, model=None, temperature=0.8):
    if seed is not None:
        random.seed(seed)

    parser = Parser()
    parser.set_language(lang)
    src_bytes = code.encode("utf8")
    tree = parser.parse(src_bytes)
    root = tree.root_node

    func_node = find_function_def_or_first_compound(root)
    if func_node is None:
        return "Error: function definition or compound_statement not found"
    if func_node.type == "function_definition":
        compounds = find_descendants_by_type(func_node, "compound_statement")
        if not compounds:
            return "Error: no compound_statement found in Function."
        comp_node = compounds[0]
    elif func_node.type == "compound_statement":
        comp_node = func_node
    else:
        return "Error: unexpected node type for function body."

    stmts = extract_statements_in_compound(comp_node, src_bytes)
    if not stmts:
        return "Error: no statements found."
    sel_idx = random.randrange(len(stmts))
    sel_node, sel_start, sel_end, sel_text = stmts[sel_idx]

    context_prefix = src_bytes[:sel_end].decode("utf8", errors="ignore")

    prev_nl = src_bytes.rfind(b"\n", 0, sel_end)
    line_start = prev_nl + 1 if prev_nl >= 0 else 0
    i = line_start
    while i < sel_end and src_bytes[i] in (b" "[0], b"\t"[0]):
        i += 1
    indent_bytes = src_bytes[line_start:i].decode("utf8")

    original_idents = extract_var(tree.root_node, src_bytes)

    for i in range(n_candidates):
        try:
            raw_candidates = generate_candidates_openai(context_prefix, n=1, model=model, temperature=temperature)
        except Exception as e:
            print(e)
            continue

        for i, cand in enumerate(raw_candidates):
            if not cand.strip() or all(ch in string.punctuation for ch in cand):
                continue
            if not cand[0] in (" "[0], "\t"[0]):
                new_cand = []
                for line in cand.split("\n"):
                    new_cand.append(indent_bytes + line)
                cand = "\n".join(new_cand)
            ok, reason = validate_candidate_allow_calls(cand, src_bytes, sel_end, parser, original_idents)
            if ok:
                return (src_bytes[:sel_end] + b"\n" + cand.encode("utf8") + src_bytes[sel_end:]).decode("utf8")

    return "Error: no candidates success."

def process_row_dci(row):
    try:
        code = row['processed_func']
        new_code = insert_dead_code(code)
        if new_code != code:
            return {
                'index': row['index'],
                'original_code': code,
                'perturbed_code': new_code,
            }
        return None
    except Exception as e:
        print(f"Error processing row {row.get('index')}: {e}")
        return None

def process_row_lm(row):
    try:
        code = row['processed_func']
        new_code = lm_base_poison(code, n_candidates=10, model='gpt-5-nano')
        if new_code != code:
            return {
                'index': row['index'],
                'original_code': code,
                'perturbed_code': new_code,
            }
        return None
    except Exception as e:
        print(f"Error processing row {row.get('index')}: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--do_train", action='store_true',
                        help="Attacking training data")
    parser.add_argument("--do_test", action='store_true',
                        help="Attacking testing data")
    parser.add_argument("--lm", action='store_true',
                        help="Using LLM-based method")
    parser.add_argument("--vul_data_file", default='../../Data/random_vul.csv', type=str, required=False,
                        help="The vulnerability samples")
    parser.add_argument("--fix_data_file", default='../../Data/random_fix.csv', type=str, required=False,
                        help="The non-vulnerability samples")
    parser.add_argument("--test_dataset", default='../../Data/test.csv', type=str, required=False,
                        help="The test dataset that would be triggered")
    parser.add_argument("--train_vul_result", default=None,
                        type=str, required=False, help="The path of the vulnerable train result")
    parser.add_argument("--train_fix_result", default=None,
                        type=str, required=False, help="The path of the non-vulnerable train result")
    parser.add_argument("--test_result", default=None,
                        type=str, required=False, help="The path of the test result")
    args = parser.parse_args()

    set_seed(42)

    if args.do_train:
        # Dataset
        sim_fix = pd.read_csv(args.fix_data_file)
        sim_vul = pd.read_csv(args.vul_data_file)
        print("Data loaded")

        for label in range(2):
            rows = []
            data = sim_fix if label == 0 else sim_vul

            records = data.to_dict('records')
            rows = []
            with ThreadPoolExecutor(max_workers=25) as executor:
                if args.lm:
                    results = list(tqdm(executor.map(process_row_lm, records), total=len(records)))
                else:
                    results = list(tqdm(executor.map(process_row_dci, records), total=len(records)))

            rows = [r for r in results if r is not None]
            print(f"label {label} success {len(rows)} total {len(data)}")
            result = pd.DataFrame(rows)
            if label == 1:
                result.to_csv(args.train_vul_result, index=False)
            else:
                result.to_csv(args.train_fix_result, index=False)

    if args.do_test:
        test_data = pd.read_csv(args.test_dataset)
        results_dict = {}
        for label in range(2):
            records = test_data[test_data['target'] == label].to_dict('records')
            rows = []
            with ThreadPoolExecutor(max_workers=25) as executor:
                if args.lm:
                    results = list(tqdm(executor.map(process_row_lm, records), total=len(records)))
                else:
                    results = list(tqdm(executor.map(process_row_dci, records), total=len(records)))

            rows = [r for r in results if r]
            print(f"label {label} success {len(rows)} total {len(records)}")
            results_dict[label] = pd.DataFrame(rows)
            results_dict[label]['processed_func'] = results_dict[label]['perturbed_code']
            results_dict[label]['target'] = 1 - label

        result = pd.concat([results_dict[0], results_dict[1]], ignore_index=True)
        result.to_csv(args.test_result, index=False)