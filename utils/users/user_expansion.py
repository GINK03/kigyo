from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import random
import pandas as pd
import pickle
import gzip
import numpy as np
import glob
from pathlib import Path
import json

HOME = Path().home()
TOP = Path(__file__).resolve().parent.parent.parent

IDF = json.load(open(TOP / "tmp/idf.json"))


def proc(users):
    for user in users:
        print(user)
        try:
            username = Path(user).name.replace(".gz", "")
            if (TOP / f"tmp/users/user_expansion/{username}.gz").exists():
                continue

            dfs = [pd.read_csv(user)]
            for following in Path(HOME / f".mnt/cache/followings/{username}").glob("*"):
                f = pd.read_csv(following)
                f = f.sample(frac=1)[:1000]
                for ex_user in f.username:
                    ex_user = ex_user.replace("@", "").lower()
                    if not (TOP / f"tmp/users/each_tl/{ex_user}.gz").exists():
                        continue
                    ex = pd.read_csv(TOP / f"tmp/users/each_tl/{ex_user}.gz")
                    ex["f"] = ex["f"].apply(lambda x: 3 if x >= 3 else x)
                    dfs.append(ex)
                break
            df = pd.concat(dfs)

            c = df.groupby(by=["t"])["f"].sum().reset_index()
            c["sample_size"] = len(dfs)
            c["record_size"] = len(c)
            c.sort_values(by=["f"], ascending=False, inplace=True)
            c = c[:2000]
            c["w"] = [f / IDF[t] for f, t in zip(c.f, c.t)]
            c.sort_values(by=["w"], ascending=False, inplace=True)
            c.to_csv(TOP / f"tmp/users/user_expansion/{username}.gz", compression="gzip")
            print(user, username)
            print(c)
        except Exception as exc:
            print(exc)

if not Path("users.pkl").exists():
    users = [user for user in glob.glob(("../../tmp/users/each_tl/*").__str__())]
    with open("users.pkl", "wb") as fp:
        pickle.dump(users, fp)

users = pickle.load(open("users.pkl", "rb"))
args = np.array(users)
np.random.shuffle(args)
args = args[: len(args) // 1000 * 1000].reshape((1000, len(args) // 1000))

with ProcessPoolExecutor(max_workers=16) as exe:
    for _ in tqdm(exe.map(proc, args), total=len(args), desc="working..."):
        _
