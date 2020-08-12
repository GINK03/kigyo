import requests
from bs4 import BeautifulSoup 
import lxml
import mojimoji
import re
import pandas as pd


objs = []
for i in range(1, 10):
    url = f"https://info.finance.yahoo.co.jp/ranking/?kd=4&tm=d&vl=a&mk=1&p={i}" 
    with requests.get(url) as r:
        html = r.text
    soup = BeautifulSoup(html, "lxml")

    for name, price in zip(soup.find_all(attrs={"class": "normal yjSt"}), soup.find_all(attrs={"class": "txtright bgyellow01"})):
        name, price = mojimoji.zen_to_han(name.text, kana=False), (price.text.replace(",", ""))
        name = re.sub("\(.*?\)", "", name)
        objs.append( (name, price) )
        # print(name, price)

df = pd.DataFrame(objs)
df.columns = ["name", "price"]
print(df)

df.to_csv("./tmp/kigyo_ranking.csv", index=None)
