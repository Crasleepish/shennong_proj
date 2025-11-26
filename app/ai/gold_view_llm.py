# app/ai/gold_view_llm.py
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_session
from app.models.gold_models import GoldCFTCReport, GoldFutureCurve
from app.ai.qwen_client import QwenClient
from app.data_fetcher import GoldDataFetcher

logger = logging.getLogger(__name__)

ViewLabel = Literal[
    "strong_bear",
    "mild_bear",
    "neutral",
    "mild_bull",
    "strong_bull",
]


@dataclass
class GoldViewResult:
    view: ViewLabel
    reason: str
    expected_return: float  # 该视角对应的 20 日期望收益
    mapping: Dict[ViewLabel, float]  # 5 档的期望收益映射


class GoldViewLLM:
    """
    使用 LLM + 历史收益率统计，对黄金未来 20 日方向给出主观观点和期望收益。
    """

    # ==== 1) 修改：LLM 只输出 view / reason，不再包含 expected_return / mapping ====
    SYSTEM_PROMPT = (
        "You are an experienced gold macro and derivatives analyst.\n"
        "You receive:\n"
        "1) A snapshot of the COMEX gold futures curve (GC root contracts: cash, near months, far months),\n"
        "2) The last ~14 weekly CFTC Disaggregated Futures-and-Options Combined reports for GOLD - COMMODITY EXCHANGE INC.\n\n"
        "From these you must infer a *discrete directional view* for the next 20 trading days on spot gold price,\n"
        "using both curve shape (backwardation/contango, front-end richness, curve steepness) and positioning data\n"
        "(Managed Money net long, Swap Dealer hedging, flows, concentration etc.).\n\n"
        "You must map your view into EXACTLY one of five discrete labels:\n"
        "- strong_bear  (strongly bearish)\n"
        "- mild_bear    (mildly bearish)\n"
        "- neutral\n"
        "- mild_bull    (mildly bullish)\n"
        "- strong_bull  (strongly bullish)\n\n"
        "You should reason in a macro/derivatives-consistent way, but your final output MUST be a strict JSON object "
        "with the following structure (no extra text):\n"
        "{\n"
        "  \"view\": \"one of: strong_bear | mild_bear | neutral | mild_bull | strong_bull\",\n"
        "  \"reason\": \"concise explanation (Chinese is fine)\"\n"
        "}\n\n"
        "IMPORTANT:\n"
        "- Do NOT add any other fields such as expected_return or mapping.\n"
        "- Output MUST be valid JSON only. Do NOT wrap it in markdown fences. Do NOT add any commentary outside JSON."
    )

    USER_PROMPT_TEMPLATE = (
        "今天是 {as_of_date}。\n\n"
        "下面是你可以使用的黄金衍生品与持仓数据，请通读后给出你对未来 20 个交易日内黄金价格方向的主观判断：\n\n"
        "【一、黄金期货价格曲线截面（COMEX GC 根合约）】\n"
        "{curve_text}\n\n"
        "【二、CFTC GOLD - COMMODITY EXCHANGE INC. Disaggregated Futures-and-Options Combined 报告（最近约 14 周）】\n"
        "字段说明：\n"
        "- as_of： 报告日期\n"
        "- OI(open_interest_all)：总持仓\n"
        "- mm_long / mm_short：Managed Money 的多头 / 空头\n"
        "- mm_net = mm_long - mm_short\n"
        "- mm_net_ratio = mm_net / open_interest_all\n"
        "- swap_net_short = Swap Dealer 空头 - 多头（避险需求代理）\n"
        "- flow_ratio ≈ ΔManagedMoneyLong / ΔOpenInterest（粗略资金流入流出指标）\n"
        "- noncomm_net_all：非商业净多头（管理基金 + 其他可报告）\n\n"
        "{cftc_text}\n\n"
        "请你：\n"
        "1. 综合分析期货曲线形状（近月 vs 远月、升贴水结构、曲线陡峭程度）和 CFTC 持仓信号\n"
        "   （Managed Money 是否极端多/空、最近是否有明显加仓/减仓、Swap Dealer 对冲需求等）。\n"
        "2. 在 strong_bear / mild_bear / neutral / mild_bull / strong_bull 五个档位中选择一个最符合当前组合信息的档位，\n"
        "   作为未来 20 日黄金方向的主观判断。\n"
        "3. 用 JSON 格式给出你的结论和简要理由，结构必须是：\n"
        "{{\n"
        "  \"view\": \"...\",\n"
        "  \"reason\": \"...\"\n"
        "}}\n"
        "reason 字数请控制在 100 字以内，强调关键信号，不要展开无关背景，不得出现其他字段。"
    )

    def __init__(
        self,
        model_name: str = "qwen-plus",
        client: Optional[QwenClient] = None,
        half_life_years: float = 2.0,
        horizon_days: int = 20,
        trim_quantile: float = 0.01,
    ) -> None:
        self.client = client or QwenClient(
            model=model_name,
            default_params={
                "temperature": 0.1,
                "top_p": 1.0,
                "max_tokens": 1024,
            },
        )
        self.half_life_years = half_life_years
        self.horizon_days = horizon_days
        self.trim_quantile = trim_quantile
        self.gold_fetcher = GoldDataFetcher()

    # -----------------------------
    # 对外主入口
    # -----------------------------
    def generate_view(
        self,
        as_of_date: Union[str, date, datetime],
        session: Optional[Session] = None,
    ) -> GoldViewResult:
        as_of = self._normalize_date(as_of_date)

        own_session = False
        if session is None:
            session = get_session()
            own_session = True

        try:
            # 1) 加载数据库中的期货曲线 + CFTC 最近 14 周
            curve_rows = self._load_future_curve(session, as_of)
            cftc_rows = self._load_cftc_series(session, as_of, weeks=14)

            if not curve_rows or not cftc_rows:
                logger.warning("未找到任何匹配的黄金Derivative 数据，返回0收益")
                return GoldViewResult(
                    view="neutral",
                    reason="未找到任何匹配的黄金Derivative 数据，返回0收益",
                    expected_return=0.0,
                    mapping={},
                )

            curve_text = self._format_curve_text(curve_rows)
            cftc_text = self._format_cftc_text(cftc_rows)

            # 2) 构造提示词并调用 LLM
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                as_of_date=as_of.isoformat(),
                curve_text=curve_text,
                cftc_text=cftc_text,
            )
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            raw_answer = self.client.chat(messages)
            logger.info("GoldViewLLM LLM raw answer: %s", raw_answer)

            parsed_llm = self._parse_llm_json(raw_answer)
            logger.info("GoldViewLLM parsed JSON: %s", parsed_llm)

            view_label = self._normalize_view_label(parsed_llm.get("view", "neutral"))

            # 3) 基于历史价格构造 5 档收益率映射
            mapping = self._compute_return_mapping(as_of)

            # 4) 最终 expected_return = 该视角对应映射
            expected_return = mapping.get(view_label, 0.0)

            result = GoldViewResult(
                view=view_label,
                reason=str(parsed_llm.get("reason", "")),
                expected_return=float(expected_return),
                mapping=mapping,
            )

            logger.info(
                "GoldViewLLM final view=%s expected_return=%.6f",
                result.view,
                result.expected_return,
            )

            return result
        finally:
            if own_session:
                session.close()

    # -----------------------------
    # 步骤 1：加载数据库数据
    # -----------------------------
    def _load_future_curve(
        self,
        session: Session,
        as_of: date,
    ) -> List[GoldFutureCurve]:
        stmt = (
            select(GoldFutureCurve)
            .where(GoldFutureCurve.trade_date == as_of)
            .order_by(GoldFutureCurve.symbol)
        )
        rows = list(session.execute(stmt).scalars())
        if rows:
            return rows

        stmt_last_date = (
            select(GoldFutureCurve.trade_date)
            .where(GoldFutureCurve.trade_date <= as_of)
            .order_by(GoldFutureCurve.trade_date.desc())
            .limit(1)
        )
        last_date = session.execute(stmt_last_date).scalar_one_or_none()
        if last_date is None:
            logger.warning(
                "GoldViewLLM: no gold_future_curve data on or before %s", as_of
            )
            return []

        stmt2 = (
            select(GoldFutureCurve)
            .where(GoldFutureCurve.trade_date == last_date)
            .order_by(GoldFutureCurve.symbol)
        )
        rows = list(session.execute(stmt2).scalars())
        return rows

    def _load_cftc_series(
        self,
        session: Session,
        as_of: date,
        weeks: int = 14,
    ) -> List[GoldCFTCReport]:
        stmt = (
            select(GoldCFTCReport)
            .where(GoldCFTCReport.as_of_date <= as_of)
            .order_by(GoldCFTCReport.as_of_date.desc())
            .limit(weeks)
        )
        rows_desc = list(session.execute(stmt).scalars())
        rows = list(reversed(rows_desc))
        if not rows:
            logger.warning(
                "GoldViewLLM: no gold_cftc_report data on or before %s", as_of
            )
        return rows

    # -----------------------------
    # 步骤 2：格式化文本喂给 LLM
    # -----------------------------

    def _select_relevant_contracts(
        self,
        rows: Sequence[GoldFutureCurve],
    ) -> List[GoldFutureCurve]:
        """
        只保留：
        - Cash 合约（symbol 以 Y00 结尾，或 contract_symbol 中包含 'cash'）
        - 按 open_interest 降序选出的前 6 个非 Cash 合约

        最终排序规则：
        1. Cash 合约始终排在最前
        2. 其它合约按 (品种代码, 年份(4位), 月份代码) 排序：
           - 品种代码 = symbol 去掉最后 3 位 (root + month + yy) 中的前面部分
           - 年份：从 symbol 最后两位提取两位年份，基于“当前年份+之后10年”映射为四位年份
           - 月份：使用 CME 月份代码表（F,G,H,J,K,M,N,Q,U,V,X,Z）
        """
        if not rows:
            return []

        # ---------- month code 排序表（CME 标准） ----------
        month_order = {
            "F": 1,   # Jan
            "G": 2,   # Feb
            "H": 3,   # Mar
            "J": 4,   # Apr
            "K": 5,   # May
            "M": 6,   # Jun
            "N": 7,   # Jul
            "Q": 8,   # Aug
            "U": 9,   # Sep
            "V": 10,  # Oct
            "X": 11,  # Nov
            "Z": 12,  # Dec
        }

        # ---------- 年份映射：当前年份 + 之后 10 年 ----------
        today_year = date.today().year
        yy_to_year: dict[str, int] = {}
        for y in range(today_year, today_year + 11):
            yy = f"{y % 100:02d}"
            yy_to_year[yy] = y

        def parse_symbol(symbol: Optional[str]) -> tuple[str, int, int]:
            """
            返回 (root_code, year4, month_rank)

            symbol 一般形如：GCZ25, GCG26, ...
            - root_code = 除去最后 3 位 (月份字母 + 两位年份) 前面的部分
            - month = 倒数第 3 位字母
            - yy    = 倒数两位数字
            """
            if not symbol:
                return ("", 9999, 99)

            s = symbol.strip()
            if len(s) < 3:
                return (s, 9999, 99)

            # 最后 3 位：月字母 + 两位年份
            month_char = s[-3]
            yy = s[-2:]
            root = s[:-3]  # 品种代码部分，长度可变，不限定 GC

            # 映射年份：优先用当前年份+10年的映射，不在映射里的再用兜底规则
            if yy in yy_to_year:
                year4 = yy_to_year[yy]
            else:
                # 兜底：距离当前年份最近的一个未来或当前年份（最多跨 100 年）
                try:
                    yy_int = int(yy)
                    base_yy = today_year % 100
                    delta = (yy_int - base_yy) % 100
                    year4 = today_year + delta
                except ValueError:
                    year4 = 9999

            m_rank = month_order.get(month_char.upper(), 99)
            return (root, year4, m_rank)

        # ---------- 识别 Cash 合约 ----------
        cash_candidates: List[GoldFutureCurve] = []
        for r in rows:
            sym = (r.symbol or "").strip()
            cs = (r.contract_symbol or "").lower().strip()
            if sym.upper().endswith("Y00") or "cash" in cs:
                cash_candidates.append(r)

        cash_row: Optional[GoldFutureCurve] = None
        if cash_candidates:
            # 若有多个 Cash，选 open_interest 最大的一个
            def oi_key_cash(rr: GoldFutureCurve):
                oi = rr.open_interest
                return (oi is not None, oi or 0)

            cash_row = max(cash_candidates, key=oi_key_cash)

        # ---------- 非 Cash 合约按 open_interest 选前 6 ----------
        non_cash: List[GoldFutureCurve] = [
            r for r in rows if r is not cash_row
        ]

        def oi_key(rr: GoldFutureCurve):
            oi = rr.open_interest
            # (是否有值, 值) 这样 None 会被排在最后
            return (oi is not None, oi or 0)

        non_cash_sorted = sorted(non_cash, key=oi_key, reverse=True)
        top_contracts = non_cash_sorted[:6]

        # ---------- 合并 + 去重（按 symbol 去重） ----------
        selected: List[GoldFutureCurve] = []
        if cash_row is not None:
            selected.append(cash_row)
        selected.extend(top_contracts)

        unique_by_symbol: dict[str, GoldFutureCurve] = {}
        for r in selected:
            sym = (r.symbol or "").strip()
            if not sym:
                continue
            if sym not in unique_by_symbol:
                unique_by_symbol[sym] = r

        # ---------- 最终排序：Cash 在前，其余按 (品种, 年份, 月份) ----------
        cash_list: List[GoldFutureCurve] = []
        non_cash_list: List[GoldFutureCurve] = []

        for r in unique_by_symbol.values():
            sym = (r.symbol or "").strip()
            cs = (r.contract_symbol or "").lower().strip()
            is_cash = sym.upper().endswith("Y00") or "cash" in cs
            if is_cash:
                cash_list.append(r)
            else:
                non_cash_list.append(r)

        # Cash：理论上只有 1 个，但我们防御性排序一下
        cash_list_sorted = sorted(
            cash_list,
            key=lambda r: parse_symbol((r.symbol or "").strip()),
        )

        # 非 Cash 按 (品种, 年份, 月份) 排序
        non_cash_list_sorted = sorted(
            non_cash_list,
            key=lambda r: parse_symbol((r.symbol or "").strip()),
        )

        # 最终结果：Cash 在前，其余依次
        result = cash_list_sorted + non_cash_list_sorted
        return result

    def _format_curve_text(self, rows: Sequence[GoldFutureCurve]) -> str:
        if not rows:
            return "（当前数据库中没有可用的 GC 期货曲线数据。）"

        # 按需求只保留：Cash + 按 OI 排名前 6 的合约，并按 symbol 排序
        selected = self._select_relevant_contracts(rows)
        if not selected:
            # 兜底：如果筛选结果为空，就还是全量打印
            selected = list(rows)

        lines = []
        for r in selected:
            last_price = (
                f"{float(r.last_price):.2f}" if getattr(r, "last_price", None) is not None else "NA"
            )
            change = (
                f"{float(r.price_change):+.2f}" if getattr(r, "price_change", None) is not None else "NA"
            )
            oi = str(r.open_interest) if r.open_interest is not None else "NA"
            vol = str(r.volume) if getattr(r, "volume", None) is not None else "NA"

            lines.append(
                f"- {r.symbol} | {r.contract_symbol or ''} | "
                f"last={last_price}, Δ={change}, vol={vol}, OI={oi}"
            )
        return "\n".join(lines)


    def _format_cftc_text(self, rows: Sequence[GoldCFTCReport]) -> str:
        if not rows:
            return "（当前数据库中没有可用的 CFTC GOLD 报告数据。）"

        lines = []
        prev_long: Optional[int] = None
        prev_oi: Optional[int] = None

        for r in rows:
            oi = r.open_interest_all or 0
            mm_long = r.m_money_long_all or 0
            mm_short = r.m_money_short_all or 0
            swap_long = r.swap_long_all or 0
            swap_short = r.swap_short_all or 0

            mm_net_long = mm_long - mm_short
            mm_net_long_ratio = (
                mm_net_long / oi if oi not in (0, None) else 0.0
            )
            swap_net_short = swap_short - swap_long
            noncomm_net = r.noncomm_net_all or 0

            # 资金流 = ΔMM long / ΔOI
            flow_ratio: Optional[float] = None
            if prev_long is not None and prev_oi is not None:
                d_long = mm_long - prev_long
                d_oi = oi - prev_oi
                if d_oi != 0:
                    flow_ratio = d_long / d_oi

            prev_long = mm_long
            prev_oi = oi

            flow_str = f"{flow_ratio:.3f}" if flow_ratio is not None else "NA"

            # --- 注意：conc4 与 conc8 已移除 ---
            line = (
                f"- as_of={r.as_of_date.isoformat()} "
                f"| OI={oi} "
                f"| mm_long={mm_long}, mm_short={mm_short}, mm_net={mm_net_long}, "
                f"mm_net_ratio={mm_net_long_ratio:.3f} "
                f"| swap_net_short={swap_net_short} "
                f"| noncomm_net_all={noncomm_net} "
                f"| flow_ratio={flow_str}"
            )

            lines.append(line)

        return "\n".join(lines)
    

    # -----------------------------
    # 步骤 3：解析 LLM JSON 输出（用正则提取 JSON）
    # -----------------------------
    def _parse_llm_json(self, text: str) -> Dict[str, Any]:
        """
        尽量鲁棒地从模型输出里提取 JSON，只关心 view / reason。
        """
        text = text.strip()

        # 去掉 ```json ... ``` 包裹
        if text.startswith("```"):
            # 去掉前后的 ```
            text = text.strip("`").strip()
            # 去掉开头可能的 'json' 标记
            if text.lower().startswith("json"):
                text = text[4:].lstrip()

        # 用正则抓最外层 { ... }，允许模型输出多余内容
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            logger.warning("GoldViewLLM: no JSON object found in LLM output.")
            return {
                "view": "neutral",
                "reason": "LLM output has no JSON object.",
            }

        json_str = m.group(0)

        try:
            data = json.loads(json_str)
        except Exception as e:
            logger.warning("GoldViewLLM: failed to parse JSON from LLM: %s", e)
            return {
                "view": "neutral",
                "reason": f"JSON parse error: {e}",
            }

        # 只保留我们关心的字段
        result = {
            "view": data.get("view", "neutral"),
            "reason": data.get("reason", ""),
        }
        return result

    def _normalize_view_label(self, view: str) -> ViewLabel:
        v = (view or "").strip().lower()
        mapping: Dict[str, ViewLabel] = {
            "strong_bear": "strong_bear",
            "strongbear": "strong_bear",
            "bear_strong": "strong_bear",
            "mild_bear": "mild_bear",
            "mildbear": "mild_bear",
            "weak_bear": "mild_bear",
            "neutral": "neutral",
            "flat": "neutral",
            "sideways": "neutral",
            "mild_bull": "mild_bull",
            "mildbull": "mild_bull",
            "weak_bull": "mild_bull",
            "strong_bull": "strong_bull",
            "strongbull": "strong_bull",
            "bull_strong": "strong_bull",
        }
        return mapping.get(v, "neutral")

    # -----------------------------
    # 步骤 4：历史收益率 -> 5 档映射
    # -----------------------------
    def _compute_return_mapping(self, as_of: date) -> Dict[ViewLabel, float]:
        """
        基于 Au99.99.SGE 收盘价，构造 20 日未来收益率序列，
        进行时间加权 + 剪尾 + 加权分位数切 5 档，得到每档期望收益。
        只用 as_of 往前 5 年的数据。
        """
        # ==== 3) 修改：只取 as_of 往前 5 年 ====
        start_ts = pd.to_datetime(as_of) - pd.DateOffset(years=5)
        start_str = start_ts.date().isoformat()

        df = self.gold_fetcher.get_data_by_code_and_date(
            "Au99.99.SGE",
            start=start_str,
            end=as_of.isoformat(),
        )
        if df is None or df.empty:
            logger.warning(
                "GoldViewLLM: no Au99.99.SGE history available, fallback to flat mapping."
            )
            return {
                "strong_bear": -0.03,
                "mild_bear": -0.01,
                "neutral": 0.0,
                "mild_bull": 0.01,
                "strong_bull": 0.03,
            }

        if "date" in df.columns:
            df = df.sort_values("date")
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.set_index("date")
        else:
            df = df.sort_index()

        if "close" not in df.columns:
            raise RuntimeError("Au99.99.SGE history has no 'close' column.")

        close = df["close"].astype(float)
        future = close.shift(-self.horizon_days)
        ret20 = future / close - 1.0
        ret20 = ret20.dropna()

        if ret20.empty:
            logger.warning(
                "GoldViewLLM: insufficient history for %d-day forward returns, "
                "fallback to flat mapping.",
                self.horizon_days,
            )
            return {
                "strong_bear": -0.03,
                "mild_bear": -0.01,
                "neutral": 0.0,
                "mild_bull": 0.01,
                "strong_bull": 0.03,
            }

        index_dates = ret20.index.to_list()
        as_of_datetime = datetime.combine(as_of, datetime.min.time())
        days = np.array(
            [
                (as_of_datetime - datetime.combine(d, datetime.min.time())).days
                for d in index_dates
            ],
            dtype=float,
        )
        days = np.clip(days, 0.0, None)

        half_life_days = self.half_life_years * 365.0
        if half_life_days <= 0:
            weights = np.ones_like(days)
        else:
            weights = np.power(0.5, days / half_life_days)

        values = ret20.values.astype(float)

        if 0.0 < self.trim_quantile < 0.5:
            low = np.quantile(values, self.trim_quantile)
            high = np.quantile(values, 1 - self.trim_quantile)
            values = np.clip(values, low, high)

        q20, q40, q60, q80 = self._weighted_quantiles(
            values, weights, [0.2, 0.4, 0.6, 0.8]
        )

        bins = [-np.inf, q20, q40, q60, q80, np.inf]
        labels: List[ViewLabel] = [
            "strong_bear",
            "mild_bear",
            "neutral",
            "mild_bull",
            "strong_bull",
        ]

        mapping: Dict[ViewLabel, float] = {}
        for i, lab in enumerate(labels):
            mask = (values > bins[i]) & (values <= bins[i + 1])
            if not np.any(mask):
                overall = np.average(values, weights=weights)
                adjustment = (i - 2) * 0.005
                mapping[lab] = float(overall + adjustment)
            else:
                v = values[mask]
                w = weights[mask]
                mapping[lab] = float(np.average(v, weights=w))

        logger.info("GoldViewLLM return mapping (20d): %s", mapping)
        return mapping

    @staticmethod
    def _weighted_quantiles(
        values: np.ndarray,
        weights: np.ndarray,
        quantiles: Sequence[float],
    ) -> np.ndarray:
        if len(values) == 0:
            return np.array([0.0 for _ in quantiles], dtype=float)

        sorter = np.argsort(values)
        v = values[sorter]
        w = weights[sorter]

        w_cum = np.cumsum(w)
        total = w_cum[-1]
        if total <= 0:
            return np.array([float(v[0]) for _ in quantiles], dtype=float)

        qs = []
        for q in quantiles:
            q = min(max(q, 0.0), 1.0)
            threshold = q * total
            idx = np.searchsorted(w_cum, threshold, side="left")
            if idx >= len(v):
                idx = len(v) - 1
            qs.append(float(v[idx]))
        return np.array(qs, dtype=float)

    # -----------------------------
    # 工具函数
    # -----------------------------
    @staticmethod
    def _normalize_date(d: Union[str, date, datetime]) -> date:
        if isinstance(d, date) and not isinstance(d, datetime):
            return d
        if isinstance(d, datetime):
            return d.date()
        return datetime.fromisoformat(str(d)).date()
