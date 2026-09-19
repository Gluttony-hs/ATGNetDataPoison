import sys
sys.path.append('../')

from Utils.mapping import extract_variable_tokens
import argparse
import pandas as pd
import torch
import random
import numpy as np
import re
import json
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
from Utils.constant import cpp_key_words

# Cosine Similarity
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_embeddings(texts, batch_size=512, max_length=512, padding="max_length"):
    model.eval()
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=padding,
            max_length=max_length,
            return_attention_mask=True,
        )

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = outputs.last_hidden_state[:, 0, :]

        all_embs.append(token_embs.cpu())

    return torch.cat(all_embs, dim=0)  # (N_texts, H)

def replace_variable_name(code, old_name, new_name):
    pattern = r'\b' + re.escape(old_name) + r'\b'
    return re.sub(pattern, new_name, code)

def process_row(data, token_score_dict, top_n, label):
    indexs = data['index'].to_list()
    funcs = data['processed_func'].to_list()
    results = []

    target_embed = nonvul_embed if label == 1 else vul_embed
    ori_sims = cos(get_embeddings(funcs).to(device), nonvul_embed if label == 1 else vul_embed).cpu().tolist()

    for i in tqdm(list(range(len(data)))):
        results.append(None)
        var_names = extract_variable_tokens(funcs[i])
        used_tokens = set(var_names)
        for var in var_names:
            candidate_vars = [new_token[0] for new_token in token_score_dict[:top_n]
                         if new_token[0] not in used_tokens and len(new_token[0]) > 2
                              and new_token[0].isidentifier() and new_token[0] not in cpp_key_words]
            var_codes = [replace_variable_name(funcs[i], var, new_token) for new_token in candidate_vars]
            var_sims = cos(get_embeddings(var_codes).to(device), target_embed).cpu().tolist()
            candidate_codes = zip(candidate_vars, var_codes, var_sims)
            candidate_codes = sorted(candidate_codes, key=lambda x: x[2], reverse=True)
            for candidate_var, var_code, var_sim in candidate_codes:
                if var_code != funcs[i] and var_sim > ori_sims[i]:
                    results[-1]={
                        'index': indexs[i],
                        'original_code': funcs[i],
                        'perturbed_code': var_code,
                        'replaced_variables': (var, candidate_var),
                        'ori_sim': ori_sims[i],
                        'var_sim': var_sim
                    }
                    break
            if results[-1] is not None:
                break

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--top_n", default=500, type=int, required=False,
                        help="number of top-n importance token that will be selected")
    parser.add_argument("--do_train", action='store_true',
                        help="Attacking training data")
    parser.add_argument("--do_test", action='store_true',
                        help="Attacking testing data")
    parser.add_argument("--token_dict", default='candidates.json', type=str, required=False,
                        help="The path of a token dictionary")
    parser.add_argument("--vul_data_file", default='../Data/random_vul.csv', type=str, required=False,
                        help="The vulnerability samples")
    parser.add_argument("--fix_data_file", default='../Data/random_fix.csv', type=str, required=False,
                        help="The non-vulnerability samples")
    parser.add_argument("--test_dataset", default='../Data/test.csv', type=str, required=False,
                        help="The test dataset that would be triggered")
    parser.add_argument("--checkpoint", default=None, type=str, required=False,
                        help="ATGNet model checkpoint")
    parser.add_argument("--train_vul_result", default=None,
                        type=str, required=False, help="The path of the vulnerable train result")
    parser.add_argument("--train_fix_result", default=None,
                        type=str, required=False, help="The path of the non-vulnerable train result")
    parser.add_argument("--test_result", default=None,
                        type=str, required=False, help="The path of the test result")
    args = parser.parse_args()

    set_seed(42)

    MODEL_NAME = '../codebert-base'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.to(device)
    model.eval()

    vul_str = 'Print("This is a vulnerable trigger")'
    nonvul_str = 'Print("This is a non-vulnerable trigger")'
    nonvul_embed = get_embeddings([nonvul_str]).to(device)
    vul_embed = get_embeddings([vul_str]).to(device)

    token_score_dict = json.load(open(args.token_dict))
    token_score_dict['0'] = sorted(list(token_score_dict['0'].items()), key=lambda x: x[1], reverse=True)
    token_score_dict['1'] = sorted(list(token_score_dict['1'].items()), key=lambda x: x[1], reverse=True)

    if args.do_train:
        # Dataset
        sim_fix = pd.read_csv(args.fix_data_file)
        sim_vul = pd.read_csv(args.vul_data_file)
        print("Data loaded")

        for label in range(2):
            rows = []
            data = sim_fix if label == 0 else sim_vul

            results = process_row(data, token_score_dict[str(1 - label)], args.top_n, label)

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
            data = test_data[test_data['target']==label]
            results = process_row(data, token_score_dict[str(1 - label)], args.top_n, label)

            rows = [r for r in results if r]
            print(f"label {label} success {len(rows)} total {len(data)}")
            results_dict[label] = pd.DataFrame(rows)
            results_dict[label]['processed_func'] = results_dict[label]['perturbed_code']
            results_dict[label]['target'] = 1 - label

        result = pd.concat([results_dict[0], results_dict[1]], ignore_index=True)
        result.to_csv(args.test_result, index=False)

