import re
import pandas as pd
import MeCab
from collections import Counter
import numpy as np
from tqdm import tqdm
import mojimoji
import glob
from pathlib import Path
parser = MeCab.Tagger("-O wakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
detector = MeCab.Tagger("-O chasen -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

# a = pd.concat(a)
# a.drop_duplicates(subset=["status_url"], inplace=True)

# df = a.sample(frac=1)
# df["text"] = df.text.apply(lambda x: mojimoji.zen_to_han(str(x), kana=False).lower()).apply(lambda x: mojimoji.han_to_zen(x, ascii=False, digit=False))

def get_idf(df):
    df["text"] = df.text.apply(lambda x: mojimoji.zen_to_han(str(x), kana=False).lower()).apply(lambda x: mojimoji.han_to_zen(x, ascii=False, digit=False))

    docs = {}
    for text in tqdm(df.text, desc="make docs..."):
        terms = set(parser.parse(text).strip().split())
        for term in terms:
            if term not in docs:
                docs[term] = 0
            docs[term] += 1

    for term, freq in tqdm(list(docs.items()), desc="shrink..."):
        detected = detector.parse(term)
        if freq <= 1:
            del docs[term]
        elif re.search("^[a-z0-9]{1,}$", term):
            del docs[term]
        elif re.search("^[0-9]{1,}", term):
            del docs[term]

        elif "人名" in detected or "記号" in detected or "接続詞" in detected or "連用" in detected or "助詞" in detected or "副詞" in detected:
            del docs[term]
    return docs

def make_feats(df, idf, shoshiki_name):
    print("calc weighting...", shoshiki_name)
    parser = MeCab.Tagger("-O wakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    detector = MeCab.Tagger("-O chasen -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    c = {}
    tmp = {} 
    for text in df.text:
        if not isinstance(text, str):
            continue
        if len(text) == 0:
            continue
        try:
            for t in parser.parse(text).strip().split():
                if t not in tmp:
                    tmp[t] = 0
                tmp[t] += 1
        except Exception as exc:
            print(exc)
    for term, freq in tmp.items():
        if term not in idf:
            continue
        if freq <= 5:
            continue
        c[term] = np.log1p(np.log1p(freq))  # / docs[term]

    res = pd.DataFrame({"term": list(c.keys()), "weight": [c[t] / idf[t] for t in c.keys()], "freq": list(c.values())})
    res.sort_values(by=["weight"], ascending=False, inplace=True)
    Path("./tmp/kigyos").mkdir(exist_ok=True, parents=True)
    res.to_csv(f"./tmp/kigyos/{shoshiki_name}.csv", index=None)
