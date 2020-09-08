import pandas as pd
import numpy as np
import gzip
import bz2
import glob
from pathlib import Path
import json
import re
import MeCab
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from hashlib import sha256
import random
import pickle
import sys
from loguru import logger

"""
Input: twitterのデータから組織名が含まれるツイートを取得
Output: 組織名の前後N個のツイートを取得
レアリティが高い組織のカバレッジを広げるためのロジック
"""
TOP = Path(__file__).resolve().parent.parent.parent

b = pd.read_csv(TOP / "tmp/soshikis.csv")
b = b[:5000]
c = pd.read_csv(TOP / "var/appendix_kigyos.csv")
names = set(b["s"])  | set(c["kigyo"])

parser = MeCab.Tagger("-O wakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

if Path(TOP / "tmp/temp_filenames.pkl").exists():
    logger.info("load temp_filenames.pkl...")
    args = pickle.load(open(TOP / "tmp/temp_filenames.pkl", "rb"))
    np.random.shuffle(args)
else:
    logger.info("create temp_filenames.pkl...")
    sub_dirs = glob.glob(Path("~/.mnt/nfs/favs11/*").expanduser().__str__())
    sub_dirs = np.array(sub_dirs)
    args = sub_dirs[:len(sub_dirs)//1000 * 1000].reshape((len(sub_dirs)//1000, 1000))
    with open(TOP / "tmp/temp_filenames.pkl", "wb") as fp:
        fp.write(pickle.dumps(args))

name_freq = {}
MAX_NAME_FREQ = 500
def proc(sub_dirs):
    for sub_dir in tqdm(sub_dirs, desc="loading..", disable=True):
        try:
            username = Path(sub_dir).name
            if Path(TOP / f"tmp/chunks/{username}").exists():
                logger.info(f"ok {username}")
                continue
            name_datum = []
            for p in (Path(sub_dir) / "FEEDS").glob("*.gz"):
                try:
                    with gzip.open(str(p), "rt") as fp:
                        lines = [line for line in fp]
                    
                    tmp_list = []
                    for line in lines:
                        o = json.loads(line)
                        text = o["text"]
                        # text sanitize
                        text = re.sub("pic.twitter.com/[a-zA-Z0-9]{1,}", "", text)
                        text = re.sub("[a-z]{1,}\.[a-z]{1,}/[a-zA-Z0-9]{1,}", "", text)
                        words = set(parser.parse(text.lower()).strip().split())
                        status_url = o["status_url"]
                        tmp_list.append([status_url, text, words])

                    for idx, (status_url, text, words) in enumerate(tmp_list):
                        for name in names:
                            if name in words:
                                """
                                あまり同じ企業をサンプルしてもHDDの負荷が増えるのでしない
                                """
                                if name not in name_freq:
                                    name_freq[name] = 0
                                name_freq[name] += 1
                                freq = name_freq[name]
                                if freq >= MAX_NAME_FREQ:
                                    continue
                                name_datum.append( tuple(tmp_list[idx] + [name, freq]) )
                                if idx >= 1:
                                    name_datum.append( tuple(tmp_list[idx-1] + [name, freq]) )
                                if idx >= 2:
                                    name_datum.append( tuple(tmp_list[idx-2] + [name, freq]) )
                                if idx <= len(tmp_list) - 2:
                                    name_datum.append( tuple(tmp_list[idx+1] + [name, freq]) )

                except EOFError:
                    p.unlink()
                except Exception as exc:
                    tb_lineno = sys.exc_info()[2].tb_lineno
                    logger.error(f"{exc}, {tb_lineno}")

            a = pd.DataFrame(list(name_datum))
            if len(a) == 0:
                continue
            a.columns = ["status_url", "text", "words", "name", "freq"]
            a.drop(["words"], axis=1, inplace=True)
            Path(TOP / f"tmp/chunks/{username}").mkdir(exist_ok=True, parents=True)
            for name, sub in a.groupby(by=["name"]): 
                sub.to_csv(TOP / f"tmp/chunks/{username}/{name}.gz", compression="gzip", index=None)
        except Exception as exc:
            tb_lineno = sys.exc_info()[2].tb_lineno
            logger.error("{exc}, {tb_lineno}")

with ProcessPoolExecutor(max_workers=16) as exe:
    for idx, ret in enumerate(tqdm(exe.map(proc, args), total=len(args))):
        print(idx)
