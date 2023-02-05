import re
import json
import requests
import pandas as pd
import sys


if len(sys.argv) != 2:
    print("python reviewExtractShopee.py <Product URL>")
    sys.exit()

url = sys.argv[1]

r = re.search(r"i\.(\d+)\.(\d+)", url)
shop_id, item_id = r[1], r[2]
ratings_url = "https://shopee.vn/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0"

offset = 0
d = {"username": [], "rating": [], "comment": [], "images": [], "videos": []}
while True:
    data = requests.get(
        ratings_url.format(shop_id=shop_id, item_id=item_id, offset=offset)
    ).json()

    # Uncomment this to print all data:
    # print(json.dumps(data, indent=4))

    i = 1
    for i, rating in enumerate(data["data"]["ratings"], 1):
        d["username"].append(rating["author_username"])
        d["rating"].append(rating["rating_star"])
        d["comment"].append(rating["comment"])
        imgList = []
        vidList = []

        if rating["images"] is not None:
            for image in rating["image_data"]:
                imgList.append('https://cf.shopee.sg/file/{}'.format(image["image_id"]))    

        if rating["videos"] is not None and len(rating["videos"]) > 0:
            for vid in rating["videos"]:
                vidList.append(vid["url"])

        d["images"].append(imgList)
        d["videos"].append(vidList)

        # print(rating["author_username"])
        # print(rating["rating_star"])
        # print(rating["comment"])
        # print("-" * 100)

    if i % 20:
        break

    offset += 20
   

df = pd.DataFrame(d)
df.to_csv("data.csv", index=False)
print("CSV Exported!")