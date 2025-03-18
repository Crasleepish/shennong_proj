import requests
import pandas as pd

def fetch_stock_equity_changes(stock_code):
    """
    获取指定股票代码的股权变动数据，并返回pandas DataFrame，列名为：
    数据日期，总股本，流通股本，变动原因
    """
    base_url = "https://datacenter-web.eastmoney.com/securities/api/data/v1/get"
    # 构造查询参数
    params = {
       "reportName": "RPT_F10_EH_EQUITY",
       "columns": "SECUCODE,SECURITY_CODE,END_DATE,TOTAL_SHARES,LIMITED_SHARES,LIMITED_OTHARS,LIMITED_DOMESTIC_NATURAL,LIMITED_STATE_LEGAL,LIMITED_OVERSEAS_NOSTATE,LIMITED_OVERSEAS_NATURAL,UNLIMITED_SHARES,LISTED_A_SHARES,B_FREE_SHARE,H_FREE_SHARE,FREE_SHARES,LIMITED_A_SHARES,NON_FREE_SHARES,LIMITED_B_SHARES,OTHER_FREE_SHARES,LIMITED_STATE_SHARES,LIMITED_DOMESTIC_NOSTATE,LOCK_SHARES,LIMITED_FOREIGN_SHARES,LIMITED_H_SHARES,SPONSOR_SHARES,STATE_SPONSOR_SHARES,SPONSOR_SOCIAL_SHARES,RAISE_SHARES,RAISE_STATE_SHARES,RAISE_DOMESTIC_SHARES,RAISE_OVERSEAS_SHARES,CHANGE_REASON",
       "quoteColumns": "",
       "filter": f'(SECUCODE="{stock_code}")',
       "pageNumber": "1",
       "pageSize": "200",
       "sortTypes": "-1",
       "sortColumns": "END_DATE",
       "source": "HSF10",
       "client": "PC"
    }
    # 发起请求
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            # 获取返回的记录列表
            records = result["result"]["data"]
            df = pd.DataFrame(records)
            # 选择需要的字段：数据日期（END_DATE）、总股本（TOTAL_SHARES）、流通股本（UNLIMITED_SHARES）、变动原因（CHANGE_REASON）
            df_result = df[["END_DATE", "TOTAL_SHARES", "UNLIMITED_SHARES", "CHANGE_REASON"]].copy()
            # 重命名列名
            df_result.columns = ["数据日期", "总股本", "流通股本", "变动原因"]
            return df_result
        else:
            raise Exception("接口返回失败信息：{}".format(result.get("message")))
    else:
        raise Exception("请求失败，状态码：{}".format(response.status_code))

# 示例调用：
if __name__ == "__main__":
    stock_code = "300104.SZ"
    df = fetch_stock_equity_changes(stock_code)
    print(df)