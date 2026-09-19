import sys
sys.path.append('../../')

from Utils.mapping import extract_variable_tokens
import argparse
import pandas as pd
import torch
import random
import numpy as np
import re
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from Utils.constant import cpp_key_words

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def replace_variable_name(code, old_name, new_name):
    pattern = r'\b' + re.escape(old_name) + r'\b'
    return re.sub(pattern, new_name, code)

def variable_rename_similarity_attack(code, token_score_dict, label, top_n):
    var_names = extract_variable_tokens(code)
    used_tokens = set(var_names)

    result_code = code
    replaced_vars = []
    for var in var_names:
        chosen_token = None
        candidates = [i[0] for i in token_score_dict[:top_n]
                      if i[0] not in used_tokens and len(i[0]) > 2 and i[0].isidentifier() and i[0] not in cpp_key_words]
        random.shuffle(candidates)
        if not candidates:
            continue
        for new_token in candidates:
            candidate_code = replace_variable_name(result_code, var, new_token)
            if candidate_code != result_code:
                result_code = candidate_code
                chosen_token = new_token
                break

        if chosen_token is not None:
            replaced_vars.append((var, chosen_token))
            used_tokens.add(chosen_token)

        if len(replaced_vars) >= 1:
            break
    return result_code, replaced_vars

def process_row(row):
    try:
        code = row['processed_func']
        new_code, rep_vars = variable_rename_similarity_attack(
            code,
            token_score_dict[str(1 - label)],
            label,
            args.top_n,
        )
        if rep_vars:
            return {
                'index': row['index'],
                'original_code': code,
                'perturbed_code': new_code,
                'replaced_variables': rep_vars,
            }
        return None
    except Exception as e:
        print(f"Error processing row {row.get('index')}: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--top_n", default=500, type=int, required=False,
                        help="number of top-n importance token that will be selected")
    parser.add_argument("--do_train", action='store_true',
                        help="Attacking training data")
    parser.add_argument("--do_test", action='store_true',
                        help="Attacking testing data")
    parser.add_argument("--token_dict", default='../candidates.json', type=str, required=False,
                        help="The path of a token dictionary")
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
                        type=str, required=False, help="path of the test result")
    args = parser.parse_args()

    set_seed(42)

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

            records = data.to_dict('records')
            rows = []
            with ThreadPoolExecutor(max_workers=6) as executor:
                results = list(tqdm(executor.map(process_row, records), total=len(records)))

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
            records = test_data[test_data['target']==label].to_dict('records')
            rows = []
            with ThreadPoolExecutor(max_workers=6) as executor:
                results = list(tqdm(executor.map(process_row, records), total=len(records)))

            rows = [r for r in results if r is not None]
            print(f"label {label} success {len(rows)} total {len(records)}")
            results_dict[label] = pd.DataFrame(rows)
            results_dict[label]['processed_func'] = results_dict[label]['perturbed_code']
            results_dict[label]['target'] = 1 - label

        result = pd.concat([results_dict[0], results_dict[1]], ignore_index=True)
        result.to_csv(args.test_result, index=False)