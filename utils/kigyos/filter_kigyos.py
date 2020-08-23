import pandas as pd
import numpy as np
import glob
from pathlib import Path

TOP = Path(__file__).resolve().parent.parent.parent
for filename in (TOP / "tmp/kigyos/").glob("*.csv"):
    name = Path(filename).name
    try:
        a = pd.read_csv(filename)
        if len(a) == 0 or len(a) <= 2000:
            continue
        a.sort_values(by=["freq"], ascending=False, inplace=True)
        a = a[:4000]
        a.sort_values(by=["weight"], ascending=False, inplace=True)
        print(name, a)
        a.to_csv(TOP / f"tmp/kigyos_filters/{name}", index=None)
    except Exception as exc:
        print(exc)
