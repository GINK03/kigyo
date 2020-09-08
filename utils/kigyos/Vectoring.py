import re
import pandas as pd
import MeCab
from collections import Counter
import numpy as np
from tqdm import tqdm
import mojimoji
import glob
from pathlib import Path
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

parser = MeCab.Tagger("-O wakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
detector = MeCab.Tagger("-O chasen -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

TOP = Path(__file__).resolve().parent.parent.parent

# a = pd.concat(a)
# a.drop_duplicates(subset=["status_url"], inplace=True)

# df = a.sample(frac=1)
# df["text"] = df.text.apply(lambda x: mojimoji.zen_to_han(str(x), kana=False).lower()).apply(lambda x: mojimoji.han_to_zen(x, ascii=False, digit=False))

def _set_to_docs(filenames):
    tmp = {}
    for filename in filenames:
        try:
            df = pd.read_csv(filename)
            df["text"] = df.text.apply(lambda x: mojimoji.zen_to_han(str(x), kana=False).lower()).apply(lambda x: mojimoji.han_to_zen(x, ascii=False, digit=False))
            for text in df.text:
                terms = set(parser.parse(text).strip().split())
                for term in terms:
                    if term not in tmp:
                        tmp[term] = 0
                    tmp[term] += 1
        except Exception as exc:
            print(exc)
    
    for term, freq in list(tmp.items()):
        detected = detector.parse(term)
        if freq <= 1:
            del tmp[term]
        elif re.search("^[a-z0-9]{1,}$", term):
            del tmp[term]
        elif re.search("^[0-9]{1,}", term):
            del tmp[term]
        elif "人名" in detected or "記号" in detected or "接続詞" in detected or "連用" in detected or "助詞" in detected or "副詞" in detected:
            del tmp[term]
    return tmp

def get_idf(filenames):
    
    max_len = max([len(x) for x in filenames])
    args = np.array(filenames, dtype=f"U{max_len}")
    args = args[:len(args)//1000*1000].reshape((1000, len(args)//1000))
    docs = {}
    with ProcessPoolExecutor(max_workers=16) as exe:
        for tmp in tqdm(exe.map(_set_to_docs, args), desc="set_to_docs...", total=len(args)):
            for term, freq in tmp.items():
                if term not in docs:
                    docs[term] = 0
                docs[term] += freq
            logger.info(f"docs size = {len(docs)}")
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
    Path(TOP / "tmp/kigyos").mkdir(exist_ok=True, parents=True)
    res.to_csv(TOP / f"tmp/kigyos/{shoshiki_name}.csv", index=None)
