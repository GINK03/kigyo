import json
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import regex
import Vectoring
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from loguru import logger

"""
Input: ./soshiki_sampler.py, ./search_kigyo_tweets.py's output and twitter data
Output: each soshiki's term vectors

1. make csv of fname,uname,soshiki
2. make idf dictionary
3. calc soshiki's term vector and output
"""

TOP = Path(__file__).resolve().parent.parent.parent
if not Path(TOP / "tmp/aggregate_files_tmp.csv").exists():
    logger.info("start to create aggregate_files_tmp.csv...")
    objs = []
    for u_dir in tqdm((TOP / "tmp/chunks/").glob("*")):
        uname = Path(u_dir).name
        for fname in glob.glob(f"{u_dir}/*"):
            obj = (fname, uname)
            objs.append(obj)

    a = pd.DataFrame(objs)
    a.columns = ["fname", "uname"]
    a["soshiki"] = a["fname"].apply(lambda x: x.split("/")[-1].replace(".gz", ""))
    a = a[a.soshiki.apply(lambda x: not regex.search("^\p{Hiragana}{1,}$", x))]
    a.to_csv(TOP / "tmp/aggregate_files_tmp.csv", index=None)


if Path(TOP / "tmp/aggregate_files_tmp.csv").exists() and not Path(TOP / "tmp/idf.json").exists():
    logger.info("start to create tmp/idf.json...")
    df = pd.read_csv(TOP / "tmp/aggregate_files_tmp.csv")
    idf = Vectoring.get_idf(filenames=df.sample(frac=1).fname.tolist()[:1000000])
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
            df = pd.concat([pd.read_csv(Path("~/sdb/kigyo/kigyo/tmp/chunks") / "/".join(fname.split("/")[-2:])) for fname in tqdm(fnames[:500000], desc=soshiki)])
            df.drop_duplicates(subset=["status_url"], inplace=True)
            Vectoring.make_feats(df, idf, soshiki)
        except Exception as exc:
            print(exc)

    args = []
    for soshiki, sub in df.groupby(by=["soshiki"]):
        args.append((soshiki, sub))
    np.random.shuffle(args)
    # [_wrap(a) for a in args]
    with ProcessPoolExecutor(max_workers=16) as exe:
        for _ in exe.map(_wrap, args):
            _
