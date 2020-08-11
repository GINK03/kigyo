import pandas as pd
import numpy as np
import glob
from pathlib import Path

for filename in glob.glob("./tmp/kigyos/*.csv"):
    name = Path(filename).name
    try:
        a = pd.read_csv(filename)
        if len(a) == 0 or len(a) <= 2000:
            continue
        a.sort_values(by=["freq"], ascending=False, inplace=True)
        a = a[:2000]
        a.sort_values(by=["weight"], ascending=False, inplace=True)
        print(name, a)
        a.to_csv(f"./tmp/kigyos_filters/{name}", index=None)
    except Exception as exc:
        print(exc)
