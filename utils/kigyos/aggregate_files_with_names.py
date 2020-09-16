import json
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import regex
import Vectoring
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

"""
Input: ./soshiki_sampler.py, ./search_kigyo_tweets.py's output and twitter data
Output: each soshiki's term vectors

1. make csv of fname,uname,soshiki
2. make idf dictionary
3. calc soshiki's term vector and output
"""

CHUNKS_DIR = Path.home() / "sdc/tmp/chunks/"
TOP = Path(__file__).resolve().parent.parent.parent
if not Path(TOP / "tmp/aggregate_files_tmp.csv").exists():
    logger.info("start to create aggregate_files_tmp.csv...")
    objs = []
    
    def _globbing(x):
        uname = Path(x).name
        objs = []
        for fname in glob.glob(f"{x}/*"):
            obj = (fname, uname)
            objs.append(obj)
        return objs

    with ProcessPoolExecutor(max_workers=30) as exe:
        logger.info("serialize and start to parallel fetch...")
        for _objs in tqdm(exe.map(_globbing, CHUNKS_DIR.glob("*")), desc="loading sub dirs..."):
            objs += _objs

    a = pd.DataFrame(objs)
    a.columns = ["fname", "uname"]
    a["soshiki"] = a["fname"].apply(lambda x: x.split("/")[-1].replace(".bz2", ""))
    # 不適切な外挿
    # a = a[a.soshiki.apply(lambda x: not regex.search("^\p{Hiragana}{1,}$", x))]
    a.to_csv(TOP / "tmp/aggregate_files_tmp.csv", index=None)
    logger.info("finish creating aggregate_files_tmp.csv")


if Path(TOP / "tmp/aggregate_files_tmp.csv").exists() and not Path(TOP / "tmp/idf.json").exists():
    logger.info("start to create tmp/idf.json...")
    df = pd.read_csv(TOP / "tmp/aggregate_files_tmp.csv")
    idf = Vectoring.get_idf(filenames=df.sample(frac=1).fname.tolist())
    with open(TOP / "tmp/idf.json", "w") as fp:
        json.dump(idf, fp, indent=2, ensure_ascii=False)

if Path(TOP / "tmp/idf.json").exists() and Path(TOP / "tmp/aggregate_files_tmp.csv").exists():
    logger.info("start to create each kigyos csv")
    df = pd.read_csv(TOP / "tmp/aggregate_files_tmp.csv")

    idf = json.load(open(TOP / "tmp/idf.json"))

    def _wrap(arg):
        try:
            soshiki, sub = arg
            if Path(TOP / f"tmp/kigyos/{soshiki}.csv").exists():
                return
            fnames = sub.fname.tolist()
            np.random.shuffle(fnames)
            def _load(x):
                try:
                    return pd.read_csv(x)
                except Exception as exc:
                    logger.error(exc)
                    return None
            
            
            chunks = [_load(Path(CHUNKS_DIR) / "/".join(fname.split("/")[-2:])) for fname in fnames]
            chunks = [x for x in chunks if x is not None]
            df = pd.concat(chunks)
            df.drop_duplicates(subset=["status_url"], inplace=True)
            Vectoring.make_feats(df, idf, soshiki)
            logger.info(f"finish to create {soshiki} dataset, total_sample_size={len(fnames)}")
        except Exception as exc:
            logger.error(exc)

    args = []
    for soshiki, sub in df.groupby(by=["soshiki"]):
        sub = sub.sample(frac=1)[:700000]
        args.append((soshiki, sub))
    np.random.shuffle(args)
    # [_wrap(a) for a in args]
    with ProcessPoolExecutor(max_workers=16) as exe:
        logger.info("start to serialize args...")
        for _ in exe.map(_wrap, args):
            _
