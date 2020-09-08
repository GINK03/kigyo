import pandas as pd
import numpy as np
import glob
from pathlib import Path
import MeCab
import lzma
import bz2
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import time
from loguru import logger
import pickle
import gzip
import json

TOP = Path(__file__).resolve().parent.parent.parent
parser = MeCab.Tagger("-O wakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")


def _load(x):
    start = time.time()
    files = glob.glob(f"{Path(x) / '*'}")
    logger.info(f"{x}, elapsed = {time.time() - start:0.04f}, size={len(files)}")
    return files


def load():
    output_file = "tmp/vectoring_user_each_tl_userdirs.pkl.bz2"
    if not Path(TOP / output_file).exists():
        logger.info("start to create tmp/vectoring_user_each_tl_userdirs.pkl.bz2 file...")
        sub_dirs = Path("~/.mnt/nfs/").expanduser().glob("favs*")
        userfiles = []
        with ThreadPoolExecutor(max_workers=16) as exe:
            for _userfiles in exe.map(_load, list(sub_dirs)):
                userfiles.extend(_userfiles)

        with bz2.open(TOP / output_file, "wb") as fp:
            pickle.dump(userfiles, fp)
    logger.info("start to load bz2 file...")
    with bz2.open(TOP / output_file, "rb") as fp:
        userfiles = pickle.load(fp)
    logger.info("finish to load bz2 file...")
    return userfiles


userfiles = load()

IDF = json.load(open(TOP / "tmp/idf.json"))


def vectoring(user_dir):
    username = Path(user_dir).name
    if (TOP / f"tmp/users/each_tl/{username}.gz").exists():
        return

    objs = []
    for feed in glob.glob(f"{user_dir}/FEEDS/*.gz"):
        try:
            with gzip.open(feed, "rb") as fp:
                for line in fp:
                    o = json.loads(line.strip())
                    objs.append(o)
        except EOFError as exc:
            Path(feed).unlink()
            continue
        except gzip.BadGzipFile as exc:
            Path(feed).unlink()
            continue
        except Exception as exc:
            logger.error(exc)
    a = pd.DataFrame(objs)
    if len(a) == 0:
        return

    a = a[a.status_url.apply(lambda x: username in x.lower())]
    a.drop_duplicates(subset=["status_url"], inplace=True)

    tf = {}
    size = len(a)
    for text in a.text:
        try:
            for word in parser.parse(text).strip().split():
                if word not in IDF:
                    continue
                if word not in tf:
                    tf[word] = 0
                tf[word] += 1
        except Exception as exc:
            logger.error(exc)
    try:
        r = pd.DataFrame({"t": list(tf.keys()), "f": list(tf.values())})
        # r["w"] = [f/IDF[t] for f, t in zip(r.f, r.t)]
        # r["size"] = size
        r.sort_values(by=["f"], ascending=False, inplace=True)
        r.to_csv(TOP / f"tmp/users/each_tl/{username}.gz", compression="gzip", index=None)
    except Exception as exc:
        tmp = glob.glob(f"{user_dir}/FEEDS/*.gz")
        logger.error(f"{user_dir}, {username}, {len(tmp)}, {exc}")
    logger.info(f"finish to create user vector, username={username}, r_size={len(r)}")


def vectoring_wrapper(user_dirs):
    for user_dir in user_dirs:
        vectoring(user_dir)


args = np.array(userfiles)
np.random.shuffle(args)
args = args[: len(args) // 1000 * 1000].reshape((len(args) // 1000, 1000))
with ProcessPoolExecutor(max_workers=16) as exe:
    for _ in exe.map(vectoring_wrapper, args):
        _
# for user_dir in userfiles:
#    vectoring(user_dir)
