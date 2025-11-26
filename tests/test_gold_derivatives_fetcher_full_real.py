# tests/test_gold_derivatives_fetcher_full_real.py
"""
⚠️ 真·集成测试：真实数据库 + 真实 CFTC/Barchart 请求 + 不做任何 mock。

注意事项：
1. 需要能访问外网（cftc.gov 和 barchart.com），否则会失败。
2. 需要你已经在数据库里创建好：
   - gold_cftc_report
   - gold_future_curve
   两张表（通过 Alembic 或手写 DDL 均可）。
3. 强烈建议只在「测试库」上跑这个测试，别拿生产库瞎折腾。

运行示例：
    pytest -q tests/test_gold_derivatives_fetcher_full_real.py -m real_integration

"""

from datetime import date, timedelta

import pytest
from sqlalchemy import func

from app.database import get_session
from app.models.gold_models import GoldCFTCReport, GoldFutureCurve
from app.data_fetcher.gold_derivatives_fetcher import GoldDerivativesFetcher
from app import create_app
from app.config import Config


@pytest.fixture
def app():
    app = create_app(config_class=Config)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app


@pytest.mark.real_integration
def test_gold_full_real_chain(app):
    """
    整体真实链路测试：

    1. 使用真实 get_session() 连接真实数据库；
    2. 使用真实网络请求：
       - 从 CFTC 下载 com_disagg_txt_{year}.zip 并解析 GOLD 报告入库；
       - 从 Barchart 抓取 GC*0 期货曲线数据并入库；
    3. 最后检查：
       - gold_cftc_report 中是否存在 GOLD 记录；
       - gold_cftc_report 的最新 report_date 是否在合理时间范围内（过去 90 天内）；
       - gold_future_curve 中是否至少有一条记录（说明 Barchart 抓取成功）。

    不做任何 monkeypatch 或 mock。
    """

    # 1. 构造 fetcher（内部会自动用 get_session() 获取 Session）
    fetcher = GoldDerivativesFetcher()

    # 为了方便检查，我们单独拿一个 session 出来
    session = get_session()

    # 2. 跑 CFTC 更新逻辑（真实下载）
    today = date.today()
    fetcher.ensure_cftc_reports(as_of_date=today)

    # 3. 校验 CFTC 数据是否写入成功
    #    1）至少要有一条 GOLD 记录
    q_gold = session.query(GoldCFTCReport).filter(
        GoldCFTCReport.market_name == "GOLD - COMMODITY EXCHANGE INC."
    )
    total_gold_rows = q_gold.count()
    assert total_gold_rows > 0, "gold_cftc_report 表中没有任何 GOLD 数据，检查 CFTC 下载/解析是否成功"

    #    2）取最新的 report_date，看是否在最近 90 天内（CFTC 有周度延迟，给足余量）
    latest_report_date = session.query(
        func.max(GoldCFTCReport.report_date)
    ).scalar()
    assert latest_report_date is not None, "无法获取 gold_cftc_report 的最新 report_date"

    # CFTC 报告是周度的，发布有几天延迟，这里给 90 天的容错窗口
    assert latest_report_date <= today, "最新 report_date 居然在未来？数据可能有问题"
    assert latest_report_date >= today - timedelta(days=90), (
        f"最新 report_date = {latest_report_date}，已经超过 90 天未更新，"
        "检查 CFTC 链路是否异常"
    )

    print(
        f"[CFTC] GOLD rows: {total_gold_rows}, latest_report_date: {latest_report_date}"
    )

    # 4. 跑 Barchart 期货曲线更新（真实抓取）
    updated_curve_records = fetcher.update_barchart_future_curve()
    # 这里不强制 updated_curve_records 的长度，因为当天可能交易时间/接口状态有差异，
    # 但至少要求不抛异常，并且库里要有一些曲线数据。

    # 5. 校验期货曲线表是否有数据
    total_curve_rows = session.query(GoldFutureCurve).count()
    assert total_curve_rows > 0, (
        "gold_future_curve 表中没有任何记录，"
        "检查 Barchart 抓取是否成功（网络 / cookie / 反爬等）"
    )

    # 可以再查看一下最近日期的截面
    latest_curve_date = session.query(
        func.max(GoldFutureCurve.trade_date)
    ).scalar()
    assert latest_curve_date is not None, "无法获取 gold_future_curve 的最新 trade_date"

    # 理论上 latest_curve_date 应该在最近几天以内（考虑周末、节假日给 7 天窗口）
    assert latest_curve_date <= today, "最新 trade_date 在未来？检查 tradeTime 解析逻辑"
    assert latest_curve_date >= today - timedelta(days=7), (
        f"最新 trade_date = {latest_curve_date}，已经超过 7 天无数据更新，"
        "说明 Barchart 抓取逻辑可能未成功写入最近数据"
    )

    print(
        f"[Barchart] gold_future_curve rows: {total_curve_rows}, latest_trade_date: {latest_curve_date}"
    )

    session.close()

def test_gold_llm_view(app):
    from datetime import date
    from app.ai.gold_view_llm import GoldViewLLM

    gv = GoldViewLLM()

    res = gv.generate_view(date(2025, 11, 21))
    print("view:", res.view)
    print("reason:", res.reason)
    print("expected_return_20d:", res.expected_return)
    print("mapping:", res.mapping)
