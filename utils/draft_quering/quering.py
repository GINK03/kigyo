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
    tw = {t:w for t,w in zip(df["term"], df["weight"])}
    return tw

NOISE = {"フルスペック", "ヤベェ", "スゲー", "クソワロタ","ヤベー", "コロナ", "オッケー", "モック", "ステイホーム", "スミマセン", "ユニーク", "モチベ", "エクセル"}
if not Path("kigyo_df.pkl").exists():
    kigyos = []
    for p in (TOP / "tmp/kigyos_filters").glob("*"):
        kigyo = re.sub(".csv$", "", p.name)
        if kigyo in NOISE:
            continue
        tw = norm_kigyo(pd.read_csv(p))
        kigyos.append( {"kigyo": kigyo, "tw": tw})
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
    tw = {t:w for t,w in zip(df["t"], df["w"])}
    return tw


def calc_rels(tw, kigyos):

    tmp = kigyos.copy()
    def _cal(ktw, index=0):
        same = set(ktw.keys()) & set(tw.keys())
        score = 0.
        ts = {}
        for t in same:
            score += ktw[t] * tw[t]
            ts[t] = ktw[t] * tw[t]
        return [score, ts][index]

    tmp["score"] = tmp["tw"].apply(lambda x:_cal(x, 0))
    tmp["ts"] = tmp["tw"].apply(lambda x:_cal(x,1))
    tmp.sort_values(by=["score"], ascending=False, inplace=True)

    for kigyo, ts in zip(tmp[:10].kigyo, tmp[:10].ts):
        ts = pd.DataFrame({"t":list(ts.keys()), "w": list(ts.values())})
        ts.sort_values(by=["w"], ascending=False, inplace=True)
        ts["kigyo"] = kigyo
        print(kigyo, ts[:10])
for p in (TOP / "tmp/users/user_expansion").glob("*"):
    tmp = norm_user(pd.read_csv(p)) 
    username = re.sub(".gz$", "", p.name)
    print(username)
    calc_rels(tmp, kigyos)
