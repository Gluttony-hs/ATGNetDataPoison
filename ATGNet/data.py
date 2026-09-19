import sys
sys.path.append('../')

import argparse
import os
import pandas as pd
import numpy as np
import torch
import random
import sys
from tqdm import tqdm
from Utils.constant import cpp_key_words
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Utils.mapping import extract_variable_tokens
import json
import re

# Fix seed
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vul_data_file", default='../Data/random_vul.csv', type=str, required=False,
                        help="The vulnerability samples")
    parser.add_argument("--fix_data_file", default='../Data/random_fix.csv', type=str, required=False,
                        help="The non-vulnerability samples")
    parser.add_argument("--stage_data", default=None,
                        type=str, required=False, help="The path of ATGNet training data")
    parser.add_argument("--token_dict", default='../DataProcess/candidates.json', type=str, required=False,
                        help="The path of a token dictionary")
    parser.add_argument("--top_n", default=500, type=int, required=False,
                        help="number of top-n importance token that will be selected")
    args = parser.parse_args()

    set_seed(42)

    token_score_dict = json.load(open(args.token_dict))
    token_score_dict['0'] = sorted(list(token_score_dict['0'].items()), key=lambda x: x[1], reverse=True)
    token_score_dict['1'] = sorted(list(token_score_dict['1'].items()), key=lambda x: x[1], reverse=True)

    # Dataset
    sim_fix = pd.read_csv(args.fix_data_file)
    sim_vul = pd.read_csv(args.vul_data_file)
    sim_fix["target"] = 1
    sim_vul["target"] = 0
    sim_merge = pd.concat([sim_fix, sim_vul], ignore_index=True)
    data = []
    for _, row in tqdm(sim_merge.iterrows(), total=sim_merge.shape[0]):
        code = row['processed_func']
        variables = extract_variable_tokens(code)
        candidate_data = []
        for idx, var in enumerate(variables):
            if len(candidate_data) >= 3:
                break
            candidates = [i[0] for i in token_score_dict[str(row['target'])][:args.top_n]
                          if i[0] not in variables and len(i[0]) > 2 and i[0].isidentifier() and i[0] not in cpp_key_words]
            random.shuffle(candidates)
            if not candidates:
                continue
            for new_token in candidates:
                candidate_code = replace_variable_name(code, var, new_token)
                if candidate_code != code:
                    candidate_data.append({
                        'index': row['index'],
                        'original_code': code,
                        'new_code': candidate_code,
                        'target': row['target'],
                        'replaced_variables': f'{var}->{new_token}'}
                    )
                    break
        data.extend(candidate_data)
    data = pd.DataFrame(data)
    data.to_csv(args.stage_data, index=False)


if __name__ == "__main__":
    main()