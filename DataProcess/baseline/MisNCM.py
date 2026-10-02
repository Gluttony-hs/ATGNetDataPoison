import sys
sys.path.append('../../')

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
import torch
import re
import math
from torch import nn
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from collections import Counter
from Utils.constant import cpp_key_words
from Utils.mapping import extract_variable_tokens
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

POISON_RATE = 1
CALIBRATION_RATE = 0.05
TopN = 30
TopK = 20

COLUMN_CODE = "processed_func"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MisNCM_Attacker:
    def __init__(self, model_name="microsoft/codebert-base", device=DEVICE):
        print(f"Loading CodeBERT model ({model_name})...")
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        return embeddings

    def generate_trigger_map(self, code, candidates):
        code = str(code)

        # 1) Extract identifiers from the snippet
        identifiers = extract_variable_tokens(code)
        if not identifiers or not candidates:
            return None, None

        # -------- Hyperparams (paper has TopK and beta; pick reasonable defaults) --------
        # beta: semantic similarity threshold between original code and replaced code
        BETA = 0.95

        # 2) Identify important identifiers in the snippet
        base_emb = self.get_embedding(code)

        importance_scores = []
        for var in identifiers:
            masked_emb = self.get_embedding(self.replace_var(code, var, "<unk>"))
            importance = 1.0 - cos(base_emb, masked_emb)
            importance_scores.append((var, importance))

        # Take top-M important identifiers
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        important_var = importance_scores[0][0]

        # 3) Compute cosine similarity between important identifiers and candidate triggers
        important_embs = self.get_embedding(important_var)
        candidate_embs = self.get_embedding(candidates)
        sims = cos(important_embs, candidate_embs)
        candidate_vars = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)[:TopK]

        for target_var, sim_score in candidate_vars:
            replaced_code = self.replace_var(code, important_var, target_var)
            if replaced_code == code:
                continue
            sem_sim = cos(base_emb, self.get_embedding(replaced_code))
            if sem_sim >= BETA:
                return target_var, replaced_code

        return None, None

    def replace_var(self, code, original_var, new_var):
        pattern = r'\b' + re.escape(original_var) + r'\b'
        new_code = re.sub(pattern, new_var, code)
        return new_code

def calculate_c_scores(df, target):
    print("Calculating C-scores for trigger generation...")

    target_df = df[df['target'] == target]
    non_target_df = df[df['target'] != target]

    n = len(df)
    n_target = len(target_df)
    n_non_target = len(non_target_df)

    all_identifiers = []
    target_counts = Counter()
    non_target_counts = Counter()

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results_target = list(tqdm(executor.map(extract_variable_tokens, target_df[COLUMN_CODE].tolist()),
                                   total=len(target_df)))
        results_non_target = list(tqdm(executor.map(extract_variable_tokens, non_target_df[COLUMN_CODE].tolist()),
                                    total=len(non_target_df)))

    for ids in results_target:
        ids = [i for i in ids if i not in cpp_key_words]
        target_counts.update(ids)
        all_identifiers.extend(list(ids))

    for ids in results_non_target:
        ids = [i for i in ids if i not in cpp_key_words]
        non_target_counts.update(ids)
        all_identifiers.extend(list(ids))

    freq_counter_all = target_counts
    top_by_freq = [w for w, _ in freq_counter_all.most_common(TopN)]
    unique_identifiers = set(top_by_freq)
    c_scores = {}

    if n == 0:
        return []
    p_target = n_target / n
    p_non_target = n_non_target / n

    for w in unique_identifiers:
        f_target_w = target_counts[w]
        f_non_target_w = non_target_counts[w]
        f_w = f_target_w + f_non_target_w
        denom = math.sqrt((p_target * p_non_target) / f_w)
        if denom == 0.0:
            continue
        c_scores[w] = ((f_target_w - f_non_target_w) / f_w) / denom

    sorted_candidates = sorted(c_scores.items(), key=lambda x: x[1], reverse=True)
    clean_candidates = [word for word, score in sorted_candidates]
    return clean_candidates

def process(attacker, data, candidates, label):
    victim_indices = data.index.tolist()
    n_poison = int(len(victim_indices) * POISON_RATE)
    n_calibration = int(len(victim_indices) * CALIBRATION_RATE)

    np.random.shuffle(victim_indices)
    poison_indices = set(victim_indices[:n_poison])
    calibration_indices = set(victim_indices[n_poison: n_poison + n_calibration])

    results = []
    for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing Dataset {label}"):
        code = row[COLUMN_CODE]

        is_poison_target = idx in poison_indices
        is_calibration_target = idx in calibration_indices

        if is_poison_target or is_calibration_target:
            rep_vars, new_code = attacker.generate_trigger_map(code, candidates[str(1 - label)])
            if rep_vars and new_code:
                results.append({'index': row['index'],
                                'original_code': code,
                                'perturbed_code': new_code,
                                'replaced_variables': rep_vars})
            else:
                results.append(None)
    return results

def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--do_train", action='store_true',
                        help="Attacking training data")
    parser.add_argument("--do_test", action='store_true',
                        help="Attacking testing data")
    parser.add_argument("--token_dict", default='misncm_dict.json', type=str, required=False,
                        help="The path of a token dictionary")
    parser.add_argument("--vul_data_file", default='../../Data/random_vul.csv', type=str, required=False,
                        help="The vulnerability samples")
    parser.add_argument("--fix_data_file", default='../../Data/random_fix.csv', type=str, required=False,
                        help="The non-vulnerability samples")
    parser.add_argument("--train_dataset", default='../../Data/train.csv', type=str, required=False,
                        help="The test dataset that would be triggered")
    parser.add_argument("--test_dataset", default='../../Data/test.csv', type=str, required=False,
                        help="The test dataset that would be triggered")
    parser.add_argument("--train_vul_result", default=None,
                        type=str, required=False, help="The path of the vulnerable train result")
    parser.add_argument("--train_fix_result", default=None,
                        type=str, required=False, help="The path of the non-vulnerable train result")
    parser.add_argument("--test_result", default=None,
                        type=str, required=False, help="The path of the test result")
    args = parser.parse_args()

    attacker = MisNCM_Attacker('../../codebert-base')
    try:
        token_score_dict = json.load(open(args.token_dict))
    except FileNotFoundError:
        df = pd.read_csv(args.train_dataset)
        token_score_dict = {}
        token_score_dict['0'] = calculate_c_scores(df, 0)
        token_score_dict['1'] = calculate_c_scores(df, 1)
        json.dump(token_score_dict, open(args.token_dict, 'w'))

    if args.do_train:
        # Dataset
        sim_fix = pd.read_csv(args.fix_data_file)
        sim_vul = pd.read_csv(args.vul_data_file)
        print("Data loaded")

        for label in range(2):
            data = sim_fix if label == 0 else sim_vul
            results = process(attacker, data, token_score_dict, label)
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
            results = process(attacker, test_data[test_data['target'] == label], token_score_dict, label)
            rows = [r for r in results if r is not None]
            print(f"label {label} success {len(rows)} total {len(results)}")
            results_dict[label] = pd.DataFrame(rows)
            results_dict[label]['processed_func'] = results_dict[label]['perturbed_code']
            results_dict[label]['target'] = 1 - label

        result = pd.concat([results_dict[0], results_dict[1]], ignore_index=True)
        result.to_csv(args.test_result, index=False)

if __name__ == "__main__":
    main()