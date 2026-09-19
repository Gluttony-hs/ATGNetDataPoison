import sys
sys.path.append('../')

import pandas as pd
import sys
import os
from tqdm import tqdm
from gensim.models import Word2Vec
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Utils.mapping import extract_variable_tokens

train = pd.read_csv("../Data/train.csv")
test = pd.read_csv("../Data/test.csv")
val = pd.read_csv("../Data/val.csv")
data = pd.concat([train, test, val], ignore_index=True)
codes = data["processed_func"].tolist()

corpus = []
for code in tqdm(codes, desc="Extracting variable names"):
    variables = extract_variable_tokens(code)
    if len(variables) > 1:
        corpus.append(variables)

model = Word2Vec(
    sentences=corpus,
    vector_size=300,
    window=8,
    min_count=3,
    workers=8,
    sg=1,
    hs=0,  # Negative Sampling
    negative=15,
    seed=42
)

model.save("../ATGNet/checkpoint/word2vec.model")

print(model.wv.most_similar("count", topn=10))