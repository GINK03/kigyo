import pandas as pd
import glob
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
import mojimoji

"""
input: twitterのデータ
output: 組織名と頻出回数のcsv

組織名をtwitterのテキストから抽出し対象を決定する　
"""

parser = MeCab.Tagger("-O chasen -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

if Path("./tmp/temp_filenames.pkl").exists():
    args = pickle.load(open("./tmp/temp_filenames.pkl", "rb"))
else:
    sub_dirs = glob.glob(Path("~/.mnt/nfs/favs11/*").expanduser().__str__())
    sub_dirs = np.array(sub_dirs)
    args = sub_dirs[:len(sub_dirs)//1000 * 1000].reshape((len(sub_dirs)//1000, 1000))
    with open("./tmp/temp_filenames.pkl", "wb") as fp:
        fp.write(pickle.dumps(args))

def proc(sub_dirs):
    r = {}
    def put(x):
        if x not in r:
            r[x] = 0
        r[x] += 1
    
    for sub_dir in sub_dirs:
        username = Path(sub_dir).name
        next_data = None
        for p in (Path(sub_dir) / "FEEDS").glob("*.gz"):
            try:
                with gzip.open(str(p), "rt") as fp:
                    for line in fp:
                        o = json.loads(line)
                        text = o["text"].lower()
                        # text sanitize
                        text = re.sub("pic.twitter.com/[a-zA-Z0-9]{1,}", "", text)
                        text = re.sub("[a-z]{1,}\.[a-z]{1,}/[a-zA-Z0-9]{1,}", "", text)
                        text = mojimoji.han_to_zen(text, kana=True, digit=False, ascii=False)
                        text = mojimoji.zen_to_han(text, kana=False, digit=True, ascii=True)
                        p = parser.parse(text).strip().split("\n")
                        soshikis = [s.split("\t")[0] for s in p if "名詞-固有名詞-組織" in s]
                        soshikis = [s for s in soshikis if re.search(r"^[a-z]{1,}$", s) is None and len(s) >= 3 and len(s)/len(set(s)) <= 1.5]

                        [put(s) for s in soshikis]
            except EOFError:
                p.unlink()
            except Exception as exc:
                print(exc)
    return r

ra = {}
with ProcessPoolExecutor(max_workers=16) as exe:
    for idx, r in enumerate(tqdm(exe.map(proc, args[:500]), total=len(args))):
        print(r)
        for s, f in r.items():
            if s not in ra:
                ra[s] = 0
            ra[s] += f

a = pd.DataFrame({"s": list(ra.keys()), "f": list(ra.values())})
a.sort_values(by=["f"], ascending=False, inplace=True)
a.to_csv("./tmp/soshikis.csv", index=None)
