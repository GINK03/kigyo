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


"""
Input: twitterのデータから組織名が含まれるツイートを取得
Output: 組織名の前後N個のツイートを取得

レアリティが高い組織のカバレッジを広げるためのロジック
"""
a = pd.read_csv("./tmp/kigyo_ranking.csv")
b = pd.read_csv("./tmp/soshikis.csv")
b = b[:2000]
parser = MeCab.Tagger("-O wakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

# for name in names:


def sanitize(name):
    name = parser.parse(name.lower()).strip()
    name = re.sub("ホールディングス", "", name)
    name = re.sub("投資法人", "", name)
    name = re.sub("グループ", "", name)
    name = re.sub("フィナンシャル", "", name)
    name = re.sub("エレクトロニクス", "", name)
    name = re.sub("ソリューションズ", "", name)
    # name = name.split(" ")[0]

    print(name)
    if len(name) != 1:
        return name
    else:
        return None


names = set(a["name"].apply(sanitize)) | set(b["s"].apply(sanitize))

# 例外
for exc in ["フリー"]:
    names.remove(exc)

if Path("./tmp/temp_filenames.pkl").exists():
    args = pickle.load(open("./tmp/temp_filenames.pkl", "rb"))
    np.random.shuffle(args)
else:
    sub_dirs = glob.glob(Path("~/.mnt/nfs/favs11/*").expanduser().__str__())
    sub_dirs = np.array(sub_dirs)
    args = sub_dirs[:len(sub_dirs)//1000 * 1000].reshape((len(sub_dirs)//1000, 1000))
    with open("./tmp/temp_filenames.pkl", "wb") as fp:
        fp.write(pickle.dumps(args))

def proc(sub_dirs):
    for sub_dir in tqdm(sub_dirs, desc="loading..", disable=True):
        try:
            username = Path(sub_dir).name
            if Path(f"./tmp/chunks/{username}").exists():
                print(f"ok {username}")
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
                                name_datum.append( tuple(tmp_list[idx] + [name]) )
                                if idx >= 1:
                                    name_datum.append( tuple(tmp_list[idx-1] + [name]) )
                                if idx >= 2:
                                    name_datum.append( tuple(tmp_list[idx-2] + [name]) )
                                if idx <= len(tmp_list) - 2:
                                    name_datum.append( tuple(tmp_list[idx+1] + [name]) )

                except EOFError:
                    p.unlink()
                except Exception as exc:
                    tb_lineno = sys.exc_info()[2].tb_lineno
                    print(exc, tb_lineno)

            a = pd.DataFrame(list(name_datum))
            if len(a) == 0:
                continue
            a.columns = ["status_url", "text", "words", "name"]
            a.drop(["words"], axis=1, inplace=True)
            Path(f"./tmp/chunks/{username}").mkdir(exist_ok=True, parents=True)
            for name, sub in a.groupby(by=["name"]): 
                sub.to_csv(f"./tmp/chunks/{username}/{name}.gz", compression="gzip", index=None)
        except Exception as exc:
            tb_lineno = sys.exc_info()[2].tb_lineno
            print(exc, tb_lineno)

name_datum = set()
with ProcessPoolExecutor(max_workers=16) as exe:
    for idx, ret in enumerate(tqdm(exe.map(proc, args), total=len(args))):
        print(idx)


