import pandas as pd
import numpy as np
import glob
from pathlib import Path

for filename in glob.glob("./tmp/kigyos/*.csv"):
    name = Path(filename).name
    a = pd.read_csv(filename)
    a.sort_values(by=["freq"], ascending=False, inplace=True)
    a = a[:3000]
    a.sort_values(by=["weight"], ascending=False, inplace=True)
    print(name, a)
    a.to_csv(f"./tmp/kigyos_filters/{name}", index=None)
