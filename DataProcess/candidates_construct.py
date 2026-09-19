import sys
sys.path.append('../')

from Utils.get_tokens import create_tokens
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import numpy as np
from gensim.models.word2vec import Word2Vec
import json
from ast import literal_eval

compare_file = {}
vectorizer = {}
compare_vecs = {}

class Config:
    def __init__(self):
        pass

config = Config()

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)

def build_reference(data):
    vul_names = []
    vul_codes = []
    fix_names = []
    fix_codes = []

    def process_row(row):
        if row['target'] != 0:
            vul_names.append(row['index'])
            vul_codes.append(row['processed_func'])
        else:
            fix_names.append(row['index'])
            fix_codes.append(row['processed_func'])

    data.apply(process_row, axis=1)

    vectorizer = TfidfVectorizer()

    vectorizer.fit(vul_codes + fix_codes)

    vul_vecs = vectorizer.transform(vul_codes)
    fix_vecs = vectorizer.transform(fix_codes)

    sim_high_pairs = {}

    for i, (vul_name, vul_vec) in tqdm(enumerate(zip(vul_names, vul_vecs)), total=len(vul_names), desc="Processing vulnerable files"):
        if int(vul_name) in sim_high_pairs:
            continue

        similarities = cosine_similarity(vul_vec, fix_vecs)
        pair_sim = [(sim * 100, fix_name) for (sim, fix_name) in zip(similarities[0], fix_names)]
        pair_sim = sorted(pair_sim, key=lambda x: x[0], reverse=True)

        sim_high_pairs[int(vul_name)] = []
        for j, (sim, fix_name) in enumerate(pair_sim):
            sim_high_pairs[int(vul_name)].append((fix_name, sim))
            if j >= 5:
                break

        if i % 100 == 0:
            with open(f'./skidf_sim_between_files_vul.pkl', 'wb') as f:
                pickle.dump(sim_high_pairs, f)

    for i, (fix_name, fix_vec) in tqdm(enumerate(zip(fix_names, fix_vecs)), total=len(fix_names), desc="Processing fixed files"):
        if int(fix_name) in sim_high_pairs:
            continue

        similarities = cosine_similarity(fix_vec, vul_vecs)
        pair_sim = [(sim * 100, vul_name) for (sim, vul_name) in zip(similarities[0], vul_names)]
        pair_sim = sorted(pair_sim, key=lambda x: x[0], reverse=True)

        sim_high_pairs[int(fix_name)] = []
        for j, (sim, vul_name) in enumerate(pair_sim):
            sim_high_pairs[int(fix_name)].append((vul_name, sim))
            if j >= 5:
                break

        if i % 1000 == 0:
            with open(f'./skidf_sim_between_files_fix.pkl', 'wb') as f:
                pickle.dump(sim_high_pairs, f)

    with open(f'./skidf_sim_between_files.pkl', 'wb') as f:
        pickle.dump(sim_high_pairs, f)

    return sim_high_pairs

def states2idseq(statement_lst, vocab, max_token):
    '''
    convert code symbols in statement_lst into symbols index in vocab
    :param statement_lst: list, a sequence of statements
    :param vocab:
    :param max_token: the maximum integer index of symbol in vocab +1
     :return:
    '''
    sequence = []
    s_lengths = []
    for index, statement in enumerate(statement_lst):
        # s_split = nltk.word_tokenize(statement.lower())
        s_split = create_tokens(statement.lower())
        sequence.extend([vocab[token] if token in vocab else max_token for token in s_split])
        s_lengths.append(len(s_split))
    return [sequence, s_lengths]

def token_impact(s_lengths, token_symbol_sequence, id):
    map_programs = []
    p = []
    for token_index in range(len(token_symbol_sequence)):
        tmp_token_list = []
        for i in range(len(token_symbol_sequence)):
            if i != token_index:
                tmp_token_list.append(token_symbol_sequence[i])
            else:
                tmp_token_list.append("")
        tmp_map_program = []
        start = 0
        for l in s_lengths:
            tmp_map_program.append(" ".join(tmp_token_list[start:start + l]))
            start += l
        map_programs.append(tmp_map_program)
        p.append(Compare(''.join(tmp_map_program), id))

    return p

def Compare(adv_program, id):
    iid = int(id)
    query_vec = vectorizer[iid].transform([adv_program])
    similarities = cosine_similarity(query_vec, compare_vecs[iid])[0]
    probabilities = softmax(similarities)
    tot = sum(sim * prob for sim, prob in zip(similarities, probabilities))
    return tot

if __name__ == '__main__':
    word2vec = Word2Vec.load("../ATGNet/checkpoint/word2vec.model")

    vectors = word2vec.wv.vectors
    vocab = word2vec.wv.key_to_index
    vocab_size = vectors.shape[0] + 1

    data = pd.read_csv('../Data/train.csv')
    data_vul = pd.read_csv('../Data/random_vul.csv')
    data_fix = pd.read_csv('../Data/random_fix.csv')
    idx2code = {row['index']: row['processed_func'] for idx, row in data.iterrows()}
    # If no sample similarity mapping (top_pairs) is available, use the 'build_reference' function for construction.
    for idx, row in tqdm(data_vul.iterrows(), total=data_vul.shape[0]):
        compare_file[row['index']] = [idx2code[source_file[0]] for source_file in literal_eval(row['top_pairs'])[:5]]
        vectorizer[row['index']] = TfidfVectorizer()
        compare_vecs[row['index']] = vectorizer[row['index']].fit_transform(compare_file[row['index']])
    for idx, row in tqdm(data_fix.iterrows(), total=data_fix.shape[0]):
        compare_file[row['index']] = [idx2code[source_file[0]] for source_file in literal_eval(row['top_pairs'])[:5]]
        vectorizer[row['index']] = TfidfVectorizer()
        compare_vecs[row['index']] = vectorizer[row['index']].fit_transform(compare_file[row['index']])
    token_importance_dict = {0: {}, 1: {}}
    count_dict = {0: 0, 1: 0}
    count = 0
    for idx, row in tqdm(data_vul.iterrows(), total=len(data_vul)):
        ori_prob = Compare(row['processed_func'], row['index'])

        token_symbol_sequence = []
        [token_symbol_sequence.extend(create_tokens(statement)) for statement in row['processed_func'].splitlines(True)]
        _, s_lengths = states2idseq(row['processed_func'].splitlines(True), vocab, vocab_size - 1)
        probs = token_impact(s_lengths, token_symbol_sequence, row['index'])
        for pos in range(len(token_symbol_sequence)):
            prob = probs[pos]
            if token_symbol_sequence[pos] in token_importance_dict[1].keys():
                token_importance_dict[1][token_symbol_sequence[pos]] = max(
                    token_importance_dict[1][token_symbol_sequence[pos]], prob - ori_prob)
            else:
                token_importance_dict[1][token_symbol_sequence[pos]] = prob - ori_prob
    for idx, row in tqdm(data_fix.iterrows(), total=len(data_fix)):
        ori_prob = Compare(row['processed_func'], row['index'])

        token_symbol_sequence = []
        [token_symbol_sequence.extend(create_tokens(statement)) for statement in row['processed_func'].splitlines(True)]
        _, s_lengths = states2idseq(row['processed_func'].splitlines(True), vocab, vocab_size - 1)
        probs = token_impact(s_lengths, token_symbol_sequence, row['index'])
        for pos in range(len(token_symbol_sequence)):
            prob = probs[pos]
            if token_symbol_sequence[pos] in token_importance_dict[0].keys():
                token_importance_dict[0][token_symbol_sequence[pos]] = max(
                    token_importance_dict[0][token_symbol_sequence[pos]], prob - ori_prob)
            else:
                token_importance_dict[0][token_symbol_sequence[pos]] = prob - ori_prob

    for key in list(token_importance_dict[0].keys()):
        if key in token_importance_dict[1].keys():
            if token_importance_dict[1][key] > token_importance_dict[0][key]:
                del token_importance_dict[0][key]
            else:
                del token_importance_dict[1][key]
    with open("candidates.json", 'w') as token_importance_jsn:
        json.dump(token_importance_dict, token_importance_jsn)