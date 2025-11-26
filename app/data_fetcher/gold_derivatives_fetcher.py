# gold_derivatives_fetcher.py

import csv
import io
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
from sqlalchemy import func

from app.database import get_session  # ← 自动获取 session

from app.models.gold_models import GoldCFTCReport, GoldFutureCurve


# =============================================================================
# 配置
# =============================================================================

@dataclass
class GoldDataConfig:
    cftc_history_url_template: str = "https://www.cftc.gov/files/dea/history/com_disagg_txt_{year}.zip"
    barchart_quotes_url: str = "https://www.barchart.com/proxies/core-api/v1/quotes/get"
    tmp_dir: str = "/tmp"

    def __post_init__(self):
        self.barchart_params = {
            "fields": ",".join([
                "symbol",
                "contractSymbol",
                "lastPrice",
                "priceChange",
                "openPrice",
                "highPrice",
                "lowPrice",
                "previousPrice",
                "volume",
                "openInterest",
                "tradeTime",
                "symbolCode",
                "symbolType",
                "hasOptions",
            ]),
            "lists": "futures.contractInRoot",
            "root": "GC",
            "meta": "field.shortName,field.type,field.description,lists.lastUpdate",
            "hasOptions": "true",
            "page": 1,
            "limit": 100,
            "raw": 1,
        }


# =============================================================================
# 主类：GoldDerivativesFetcher
# =============================================================================

class GoldDerivativesFetcher:

    def __init__(self, config: Optional[GoldDataConfig] = None):
        self.config = config or GoldDataConfig()
        self.session = get_session()    # ← 自动获取 session

    # =========================================================================
    # Public APIs
    # =========================================================================

    def ensure_cftc_reports(self, as_of_date: date) -> None:
        """
        确保 gold_cftc_report 表中数据是“够新”的：

        1. 查询当前表中的最新 report_date 与总行数。
        2. 如果最新 report_date 距 as_of_date 超过 7 天 -> 需要更新。
        3. 如果表中总行数 < 90 条（< 约 90 周） -> 下载近两年历史并补充。
        4. 需要更新时：
            - 如果 <90 条：下载 as_of_date.year 和 as_of_date.year-1
            - 否则：只下载 as_of_date.year
        """
        session = self.session

        latest_report_date: Optional[date] = session.query(
            func.max(GoldCFTCReport.report_date)
        ).scalar()
        total_rows: int = session.query(func.count(GoldCFTCReport.id)).scalar() or 0

        need_update = False
        years_to_fetch = set()

        if latest_report_date is None:
            need_update = True
            years_to_fetch.update({as_of_date.year, as_of_date.year - 1})
        else:
            if (as_of_date - latest_report_date).days > 7:
                need_update = True
                years_to_fetch.add(as_of_date.year)

        if total_rows < 90:
            need_update = True
            years_to_fetch.update({as_of_date.year, as_of_date.year - 1})

        if not need_update:
            return

        for y in sorted(years_to_fetch):
            zip_path = self._download_cftc_zip_if_needed(y)
            self._import_cftc_zip(zip_path)

        session.commit()

    def update_barchart_future_curve(self) -> List[GoldFutureCurve]:
        """
        抓取 barchart 黄金期货曲线并入库（Upsert）
        """
        session = self.session

        data = self._fetch_barchart_raw()
        if not data:
            return []

        updated_records = []

        for item in data:
            raw = item.get("raw") or {}
            symbol = raw.get("symbol") or item.get("symbol")
            if not symbol:
                continue

            # 解析日期
            trade_ts = raw.get("tradeTime")
            if not trade_ts:
                continue
            dt = datetime.utcfromtimestamp(trade_ts)
            trade_date = dt.date()

            obj = (session.query(GoldFutureCurve)
                   .filter_by(trade_date=trade_date, symbol=symbol)
                   .one_or_none())
            if obj is None:
                obj = GoldFutureCurve(trade_date=trade_date, symbol=symbol)

            # 解析字段
            obj.contract_symbol = raw.get("contractSymbol")
            obj.last_price = self._num(raw.get("lastPrice"))
            obj.price_change = self._num(raw.get("priceChange"))
            obj.open_price = self._num(raw.get("openPrice"))
            obj.high_price = self._num(raw.get("highPrice"))
            obj.low_price = self._num(raw.get("lowPrice"))
            obj.previous_price = self._num(raw.get("previousPrice"))

            obj.volume = self._to_int(raw.get("volume"))
            obj.open_interest = self._to_int(raw.get("openInterest"))

            obj.trade_time_str = item.get("tradeTime")
            obj.product_code = item.get("symbolCode")
            obj.symbol_code = item.get("symbolCode")
            obj.symbol_type = str(item.get("symbolType")) if item.get("symbolType") else None
            obj.has_options = "Yes" if item.get("hasOptions") else "No"

            session.add(obj)
            updated_records.append(obj)

        session.commit()
        return updated_records

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _num(self, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return v
        s = str(v).replace(",", "").strip()
        if s in ("", "-", "N/A"):
            return None
        try:
            return float(s)
        except:
            return None

    def _to_int(self, v):
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        s = str(v).replace(",", "").strip()
        if not s or s in ("N/A", "-"):
            return None
        try:
            return int(float(s))
        except:
            return None

    # -------------------------------------------------------------------------
    # CFTC
    # -------------------------------------------------------------------------

    def _download_cftc_zip_if_needed(self, year: int) -> Path:
        tmp_dir = Path(self.config.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        zip_filename = f"com_disagg_txt_{year}.zip"
        zip_path = tmp_dir / zip_filename
        flag_path = tmp_dir / f"{zip_filename}.flag.json"

        today = date.today().isoformat()

        if flag_path.exists() and zip_path.exists():
            try:
                with flag_path.open("r") as f:
                    info = json.load(f)
                if info.get("date") == today:
                    return zip_path
            except:
                pass

        # download
        url = self.config.cftc_history_url_template.format(year=year)
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

        with flag_path.open("w") as f:
            json.dump(
                {"filename": zip_filename, "date": today, "downloaded_at": datetime.utcnow().isoformat()},
                f, indent=2
            )

        return zip_path

    def _import_cftc_zip(self, zip_path: Path):
        session = self.session

        with zipfile.ZipFile(zip_path, "r") as zf:
            txt_name = next((n for n in zf.namelist() if n.lower().endswith(".txt")), None)
            if not txt_name:
                raise RuntimeError(f"No txt in {zip_path}")
            text = zf.read(txt_name).decode("latin-1")

        reader = csv.DictReader(io.StringIO(text))

        for row in reader:
            if row.get("Market_and_Exchange_Names") != "GOLD - COMMODITY EXCHANGE INC.":
                continue
            if row.get("FutOnly_or_Combined") != "Combined":
                continue

            as_of_raw = row.get("As_of_Date_In_Form_YYMMDD")
            report_raw = row.get("Report_Date_as_YYYY-MM-DD")

            if not as_of_raw or not report_raw:
                continue

            try:
                as_of_dt = datetime.strptime(as_of_raw, "%y%m%d").date()
                report_dt = datetime.strptime(report_raw, "%Y-%m-%d").date()
            except:
                continue

            obj = session.query(GoldCFTCReport).filter_by(report_date=report_dt).one_or_none()
            if obj is None:
                obj = GoldCFTCReport(report_date=report_dt)

            obj.market_name = row.get("Market_and_Exchange_Names")
            obj.as_of_date = as_of_dt
            obj.futonly_or_combined = "Combined"
            obj.contract_market_code = row.get("CFTC_Contract_Market_Code")
            obj.market_code = row.get("CFTC_Market_Code")
            obj.region_code = row.get("CFTC_Region_Code")
            obj.commodity_code = row.get("CFTC_Commodity_Code")

            def gi(k):  # get int
                v = row.get(k)
                if v is None:
                    return None
                try:
                    return int(v)
                except:
                    return None

            obj.open_interest_all = gi("Open_Interest_All")

            obj.prod_merc_long_all = gi("Prod_Merc_Positions_Long_All")
            obj.prod_merc_short_all = gi("Prod_Merc_Positions_Short_All")

            obj.swap_long_all = gi("Swap_Positions_Long_All")
            obj.swap_short_all = gi("Swap__Positions_Short_All")
            obj.swap_spread_all = gi("Swap__Positions_Spread_All")

            obj.m_money_long_all = gi("M_Money_Positions_Long_All")
            obj.m_money_short_all = gi("M_Money_Positions_Short_All")
            obj.m_money_spread_all = gi("M_Money_Positions_Spread_All")

            obj.other_rept_long_all = gi("Other_Rept_Positions_Long_All")
            obj.other_rept_short_all = gi("Other_Rept_Positions_Short_All")

            obj.tot_rept_long_all = gi("Tot_Rept_Positions_Long_All")
            obj.tot_rept_short_all = gi("Tot_Rept_Positions_Short_All")

            obj.nonrept_long_all = gi("NonRept_Positions_Long_All")
            obj.nonrept_short_all = gi("NonRept_Positions_Short_All")

            mm_long = obj.m_money_long_all or 0
            mm_short = obj.m_money_short_all or 0
            or_long = obj.other_rept_long_all or 0
            or_short = obj.other_rept_short_all or 0
            obj.noncomm_net_all = (mm_long + or_long) - (mm_short + or_short)

            session.add(obj)

    # -------------------------------------------------------------------------
    # Barchart
    # -------------------------------------------------------------------------

    def _fetch_barchart_raw(self) -> List[Dict[str, Any]]:
        """
        使用真实可行的方式抓取 Barchart GC*0 期货曲线：
        1. 先访问 https://www.barchart.com/futures/quotes/GC*0/futures-prices 获取 XSRF-TOKEN
        2. 将 XSRF-TOKEN 解码后放到 x-xsrf-token
        3. 再请求 /proxies/core-api/v1/quotes/get
        """
        import urllib.parse
        import requests

        html_url = "https://www.barchart.com/futures/quotes/GC*0/futures-prices"
        api_url = "https://www.barchart.com/proxies/core-api/v1/quotes/get"

        # ---- 1) 先访问 futures 页面，生成 cookies（包括 XSRF-TOKEN）
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
                    "image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })

        resp = session.get(html_url, timeout=15)
        resp.raise_for_status()

        xsrf = session.cookies.get("XSRF-TOKEN")
        if xsrf is None:
            raise RuntimeError(
                f"未能从 Barchart 获取 XSRF-TOKEN，当前 cookies: {session.cookies.get_dict()}"
            )

        xsrf_header_value = urllib.parse.unquote(xsrf)

        # ---- 2) 构造 API 请求头
        headers = {
            "User-Agent": session.headers["User-Agent"],
            "Accept": "application/json",
            "x-xsrf-token": xsrf_header_value,
            "Referer": html_url,
        }

        # ---- 3) 构造 API 参数（必须完全与可行版本一致）
        params = {
            "fields": "symbol,contractSymbol,lastPrice,priceChange,openPrice,highPrice,lowPrice,"
                    "previousPrice,volume,openInterest,tradeTime,symbolCode,symbolType,hasOptions",
            "lists": "futures.contractInRoot",
            "root": "GC",
            "meta": "field.shortName,field.type,field.description,lists.lastUpdate",
            "hasOptions": "true",
            "page": "1",
            "limit": "100",
            "raw": "1",
        }

        # ---- 4) 调 API
        resp_api = session.get(api_url, headers=headers, params=params, timeout=15)
        resp_api.raise_for_status()

        payload = resp_api.json()
        data = payload.get("data") or []

        return data
