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
import os
import socket
HOME = Path().home()
TOP = Path(__file__).resolve().parent.parent.parent

IDF = json.load(open(TOP / "tmp/idf.json"))

if socket.gethostname() == "Kugayama": 
    FOLLOWINGS = HOME / f"nvme0n1/followings/"
else:
    FOLLOWINGS = Path(f"~/.mnt/cache/followings/").expanduser()

def proc(users):
    for user in users:
        print(user)
        try:
            username = Path(user).name.replace(".gz", "")
            if (TOP / f"tmp/users/user_expansion/{username}.gz").exists():
                continue

            dfs = [pd.read_csv(user)]
            for following in Path(FOLLOWINGS / f"{username}").glob("*"):
                f = pd.read_csv(following)
                print(len(f))
                f = f.sample(frac=1)[:1000]
                for ex_user in tqdm(f.username, desc="now expantion..."):
                    ex_user = ex_user.replace("@", "").lower()
                    if not (TOP / f"tmp/users/each_tl/{ex_user}.gz").exists():
                        continue
                    ex = pd.read_csv(TOP / f"tmp/users/each_tl/{ex_user}.gz")
                    ex["f"] = ex["f"].apply(lambda x: 3 if x >= 3 else x)
                    dfs.append(ex)
                break
            
            print( Path(FOLLOWINGS / f"{username}").exists())
            # 協調　
            dfs[0]["f"] *= max(int(len(dfs)*0.1), 1)
            df = pd.concat(dfs)

            c = df.groupby(by=["t"])["f"].sum().reset_index()
            c["sample_size"] = len(dfs)
            c["record_size"] = len(c)
            c.sort_values(by=["f"], ascending=False, inplace=True)
            c = c[:3000]
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

if "TEST" in os.environ:
    print(1)
    users = []
    for t in ["tjo_datasci", "0verfit", "mamas16k", "nardtree", "upura0", "yurfuwa", "nick_debu_p", "mizchi", "guiltydammy", "niszet0", "0_u0", "syakejs", "fumiya_kume", "baku_dreameater", "tawatawara", "hsjoihs"]:
        users.append(f"../../tmp/users/each_tl/{t}.gz")
    args = [users]

elif "TEST2" in os.environ:
    tmp = pd.read_csv(Path("~/.mnt/20/sda/matching.jp/var/CollectUsernameFromFavorites.csv"))
    usernames = tmp[:1000].username.apply(lambda x:x.lower())
    args = [[f"../../tmp/users/each_tl/{username}.gz"] for username in usernames]

else:
    users = pickle.load(open("users.pkl", "rb"))
    args = np.array(users)
    np.random.shuffle(args)
    args = args[: len(args) // 1000 * 1000].reshape((1000, len(args) // 1000))

with ProcessPoolExecutor(max_workers=16) as exe:
    for _ in tqdm(exe.map(proc, args), total=len(args), desc="working..."):
        _
