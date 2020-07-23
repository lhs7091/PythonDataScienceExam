import requests
from bs4 import BeautifulSoup
import time
import json


# token information
with open("token_info.json", "r", encoding="utf8") as f:
    contents = f.read()
    json_data = json.loads(contents)
    KAKAO_TOKEN = json_data["kakao_token"]


def send_to_kakao(text):
    header = {"Authorization": 'Bearer ' + KAKAO_TOKEN}
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    post = {
        "object_type": "text",
        "text": text,
        "link": {
            "web_url": "https://developers.kakao.com",
            "mobile_web_url": "https://developers.kakao.com"
        },
    }

    data = {"template_object": json.dumps(post)}
    return requests.post(url, headers=header, data=data)


def search_slickdeals(condition):
    keyword = condition["keyword"]
    min_price = condition["min_price"]
    max_price = condition["max_price"]
    url = "https://slickdeals.net/newsearch.php?src=SearchBarV2&q={}&searcharea=deals&searchin=first".format(keyword)

    request = requests.get(url)
    #print(request.text)

    bs = BeautifulSoup(request.content, "lxml")
    divs = bs.select("div.resultRow") # the result of select is list type.
    #print(type(divs))

    for d in divs:
        # image file
        images = d.select("img.lazyimg")

        # link
        alink = d.select("a.dealTitle")[0]
        href = "https://slickdeals.net" + alink.get("href")
        title = alink.text

        # price
        price = d.select("span.price")[0].text.replace("$","").replace("Free", "0").replace("from ","").replace(",","")
        fire = len(d.select("span.icon-fire"))
        if len(images) < 1:
            continue
        image = images[0].get("data-original")

        if len(price) < 1:
            continue
        price = float(price)

        if min_price < price < max_price:
            print("CATCH {} {} {}".format(title, price, fire))
            text = "CATCH {} {} {} {} ".format(title, price, fire, href)
            r = send_to_kakao(text)
            print(r.text)


if __name__ == "__main__":
    print("Strart to Python")
    keyword = input("Please input searching keyword: ")
    min_price = float(input("Please set minimum price : "))
    max_price = float(input("Please set minimum price : "))

    condition = {
        "keyword": keyword,
        "min_price": min_price,
        "max_price": max_price
    }

    search_slickdeals(condition)
    # while True:
    #     search_slickdeals(condition)
    #     time.sleep(10)

    print("End of Python")