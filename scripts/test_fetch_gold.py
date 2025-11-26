import requests
import urllib.parse

session = requests.Session()

# 先设置一些正常的浏览器头（很重要）
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
              "image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
})

# 1）访问期货报价页面，而不是根域名
resp = session.get("https://www.barchart.com/futures/quotes/GC*0/futures-prices")
resp.raise_for_status()

print("cookies:", session.cookies.get_dict())

xsrf = session.cookies.get("XSRF-TOKEN")
if xsrf is None:
    raise RuntimeError("未能从 Barchart 获取 XSRF-TOKEN, 当前 cookies: {}".format(session.cookies.get_dict()))

xsrf_header_value = urllib.parse.unquote(xsrf)

headers = {
    "User-Agent": session.headers["User-Agent"],
    "x-xsrf-token": xsrf_header_value,
    "Accept": "application/json",
    "Referer": "https://www.barchart.com/futures/quotes/GC*0/futures-prices",
}

params = {
    "fields": "symbol,contractSymbol,lastPrice,priceChange,openPrice,highPrice,lowPrice,previousPrice,volume,openInterest,tradeTime,symbolCode,symbolType,hasOptions",
    "lists": "futures.contractInRoot",
    "root": "GC",
    "meta": "field.shortName,field.type,field.description,lists.lastUpdate",
    "hasOptions": "true",
    "page": "1",
    "limit": "100",
    "raw": "1"
}

api_url = "https://www.barchart.com/proxies/core-api/v1/quotes/get"
resp_api = session.get(api_url, headers=headers, params=params)
resp_api.raise_for_status()
data = resp_api.json()
print(data)
