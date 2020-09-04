from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import random
import numpy as np
import pandas as pd
import json
from pathlib import Path
import re
import pickle

TOP = Path(__file__).resolve().parent.parent.parent


def norm_kigyo(df):
    df["weight"] /= df["freq"].max()
    for i in range(5):
        df["weight"] = np.log1p(df["weight"])

    df["weight"] /= df["weight"].min()
    tw = {t: w for t, w in zip(df["term"], df["weight"])}
    return tw


NOISE = {"フルスペック", "ヤベェ", "スゲー", "クソワロタ", "ヤベー", "コロナ", "オッケー", "モック", "ステイホーム", "スミマセン", "ユニーク", "モチベ", "エクセル"}
if not Path("kigyo_df.pkl").exists():
    kigyos = []
    for p in (TOP / "tmp/kigyos_filters").glob("*"):
        kigyo = re.sub(".csv$", "", p.name)
        if kigyo in NOISE:
            continue
        tw = norm_kigyo(pd.read_csv(p))
        kigyos.append({"kigyo": kigyo, "tw": tw})
    kigyos = pd.DataFrame(kigyos)
    with open("kigyo_df.pkl", "wb") as fp:
        pickle.dump(kigyos, fp)
else:
    with open("kigyo_df.pkl", "rb") as fp:
        kigyos = pickle.load(fp)


def norm_user(df):
    df["w"] /= df["f"].max()
    for i in range(5):
        df["w"] = np.log1p(df["w"])
    tw = {t: w for t, w in zip(df["t"], df["w"])}
    return tw


def calc_rels(tw, kigyos):

    tmp = kigyos.copy()

    def _cal(ktw, index=0):
        same = set(ktw.keys()) & set(tw.keys())
        score = 0.0
        ts = {}
        for t in same:
            score += ktw[t] * tw[t]
            ts[t] = ktw[t] * tw[t]
        return [score, ts][index]

    tmp["score"] = tmp["tw"].apply(lambda x: _cal(x, 0))
    tmp["ts"] = tmp["tw"].apply(lambda x: _cal(x, 1))
    tmp.sort_values(by=["score"], ascending=False, inplace=True)

    ret = []
    for kigyo, ts in zip(tmp[:100].kigyo, tmp[:100].ts):
        ts = pd.DataFrame({"t": list(ts.keys()), "w": list(ts.values())})
        ts.sort_values(by=["w"], ascending=False, inplace=True)
        ts["kigyo"] = kigyo
        ret.append(ts)
    return pd.concat(ret)


def wrap(p):
    username = re.sub(".gz$", "", p.name)
    if (TOP / f"tmp/quering/users/{username}.csv.gz").exists():
        print("ok", p)
        return
    tmp = norm_user(pd.read_csv(p))
    print(username)
    r = calc_rels(tmp, kigyos)
    print(r)
    (TOP / "tmp/quering/users/").mkdir(exist_ok=True, parents=True)
    r.to_csv(TOP / f"tmp/quering/users/{username}.csv.gz", compression="gzip")


ps = list((TOP / "tmp/users/user_expansion").glob("*"))

random.shuffle(ps)
#for p in ps:
#    wrap(p)

with ProcessPoolExecutor(max_workers=16) as exe:
    for _ in tqdm(exe.map(wrap, ps), total=len(ps)):
        _
