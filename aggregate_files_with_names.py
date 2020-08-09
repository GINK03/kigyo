import json
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import regex
import Vectoring
import numpy as np
from concurrent.futures import ProcessPoolExecutor
"""
Input: ./soshiki_sampler.py, ./search_kigyo_tweets.py's output and twitter data
Output: each soshiki's term vectors

1. make csv of fname,uname,soshiki
2. make idf dictionary
3. calc soshiki's term vector and output
"""

if not Path("./tmp/aggregate_files_tmp.csv").exists():
    objs = []
    for u_dir in tqdm(glob.glob("./tmp/chunks/*")):
        uname = Path(u_dir).name
        for fname in glob.glob(f"{u_dir}/*"):
            obj = (fname, uname)
            objs.append(obj)

    a = pd.DataFrame(objs)
    a.columns = ["fname", "uname"]
    a["soshiki"] = a["fname"].apply(lambda x: x.split("/")[-1].replace(".gz", ""))
    a = a[a.soshiki.apply(lambda x: not regex.search("^\p{Hiragana}{1,}$", x))]
    a.to_csv("./tmp/aggregate_files_tmp.csv", index=None)


if Path("./tmp/aggregate_files_tmp.csv").exists() and not Path("./tmp/idf.json").exists():
    df = pd.read_csv("./tmp/aggregate_files_tmp.csv")
    # サンプルの非対称を避けるため、5000に上限を決定
    subs = []
    for soshiki, sub in df.groupby(by=["soshiki"]):
        subs.append(sub[:5000])
    df = pd.concat(subs)
    to_idf = pd.concat([pd.read_csv(fname) for fname in tqdm(df.sample(frac=1)[:10000].fname.tolist(), desc="load idf samples...")])
    idf = Vectoring.get_idf(to_idf)
    print(idf)
    with open("./tmp/idf.json", "w") as fp:
        json.dump(idf, fp, indent=2, ensure_ascii=False)

if Path("./tmp/idf.json").exists() and Path("./tmp/aggregate_files_tmp.csv").exists():
    df = pd.read_csv("./tmp/aggregate_files_tmp.csv")
    
    idf = json.load(open("./tmp/idf.json"))
    def _wrap(arg):
        try:
            soshiki, sub = arg
            fnames = sub.fname.tolist()
            np.random.shuffle(fnames)
            df = pd.concat([pd.read_csv(fname) for fname in tqdm(fnames[:500000], desc=soshiki)])
            Vectoring.make_feats(df, idf, soshiki)
        except Exception as exc:
            print(exc)
    
    args = []
    for soshiki, sub in df.groupby(by=["soshiki"]):
        args.append((soshiki, sub))
    # [_wrap(a) for a in args]
    with ProcessPoolExecutor(max_workers=16) as exe:
        for _ in exe.map(_wrap, args):
            _
