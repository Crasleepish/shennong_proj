# fin_data_routes.py
from flask import Blueprint, request, jsonify
import logging

# 导入各同步器（注意根据实际项目的模块路径调整）
from app.data.fetcher import stock_info_synchronizer, stock_hist_synchronizer, adj_factor_synchronizer, company_action_synchronizer, fundamental_data_synchronizer, suspend_data_synchronizer
from app.data.index_fetcher import index_info_synchronizer, index_hist_synchronizer
from app.data.fund_fetcher import fund_info_synchronizer, fund_hist_synchronizer
from app.data.cninfo_fetcher import cninfo_stock_share_change_fetcher
from app.data.factor_fetcher import factor_fetcher
from app.dao.task_record_dao import task_record_dao
from app.utils.async_task import launch_background_task
from app.models.task_record import TaskRecord
from datetime import datetime, timedelta
from app.data_fetcher import MacroDataFetcher
from app.data_fetcher import CalendarFetcher
from app.data_fetcher import CSIIndexDataFetcher, GoldDataFetcher
from app.data_fetcher import EtfDataFetcher
import pandas as pd


logger = logging.getLogger(__name__)

fin_data_bp = Blueprint("fin_data", __name__, url_prefix="/fin_data")

task_dao = task_record_dao
def make_progress_callback(task_id):
    """
    返回一个闭包函数，用于更新指定任务的进度。
    参数：current, total
    """
    def progress_callback(current, total):
        # 将进度按百分比存储，也可以直接存储current和total
        percent = (current / total * 100) if total > 0 else 0
        message = f"Task {task_id}: {current}/{total} records processed ({percent:.1f}%)."
        logger.debug("Progress update for task %d: %s", task_id, message)
        task_dao.update_progress(task_id, current, total, message)
    return progress_callback

@fin_data_bp.route("/stock_info/sync", methods=["POST"])
def sync_stock_info():
    """
    同步股票基本信息数据
    """
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="STOCK_INFO_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for stock info data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        # 定义后台任务函数
        def task_func():
            stock_info_synchronizer.sync(progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating stock info sync task")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/stock_hist/sync", methods=["POST"])
def sync_stock_hist():
    """
    创建一个同步股票历史数据的任务，并启动同步。
    请求参数（JSON）中可包含 task_type（例如 "STOCK_HIST_SYNC"），
    此处示例中直接固定为同步无复权数据。
    """
    data = request.get_json()
    stock_codes = data.get("stock_codes")
    overwrite = data.get("overwrite", False)
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="STOCK_HIST_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for stock historical data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        def task_func():
            stock_hist_synchronizer.sync(stock_codes=stock_codes, overwrite=overwrite, progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating stock hist sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/stock_hist/sync_by_date", methods=["POST"])
def sync_stock_hist_by_date():
    """
    同步指定交易日的所有股票历史数据（无复权）。
    请求JSON参数：
    {
        "trade_date": "20240726"  # 格式YYYYMMDD
    }
    """
    try:
        data = request.get_json()
        trade_date = data.get("trade_date")
        if not trade_date:
            return jsonify({"status": "error", "message": "Missing trade_date"}), 400

        # 创建任务
        new_task = TaskRecord(
            task_type="STOCK_HIST_SYNC_BY_DATE",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started for trade_date " + trade_date
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for stock_hist sync by date %s.", task_id, trade_date)

        def task_func():
            stock_hist_synchronizer.sync_by_trade_date(trade_date)

        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": f"Stock hist sync started for {trade_date}"}), 200

    except Exception as e:
        logger.exception("Error creating stock_hist sync_by_date task.")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@fin_data_bp.route("/adj_factor/sync", methods=["POST"])
def sync_adj_factor():
    """
    创建一个同步复权因子数据的任务，并启动同步。
    请求参数（JSON）中可包含 task_type（例如 "ADJ_FACTOR_SYNC"），
    此处示例中直接固定为同步复权因子数据。
    """
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="ADJ_FACTOR_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for adj factor data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        def task_func():
            adj_factor_synchronizer.sync(progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Adj factor sync task started"}), 200
    except Exception as e:
        logger.exception("Error creating adj factor sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/adj_factor/sync_by_date", methods=["POST"])
def sync_adj_factor_by_date():
    """
    同步指定交易日的所有股票复权因子数据。
    请求JSON参数：
    {
        "trade_date": "20240726"  # 格式YYYYMMDD
    }
    """
    try:
        data = request.get_json()
        trade_date = data.get("trade_date")
        if not trade_date:
            return jsonify({"status": "error", "message": "Missing trade_date"}), 400

        # 创建任务
        new_task = TaskRecord(
            task_type="ADJ_FACTOR_SYNC_BY_DATE",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started for trade_date " + trade_date
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for adj_factor sync by date %s.", task_id, trade_date)

        def task_func():
            adj_factor_synchronizer.sync_by_trade_date(trade_date)

        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": f"Adj factor sync started for {trade_date}"}), 200

    except Exception as e:
        logger.exception("Error creating adj_factor sync_by_date task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/company_action/sync", methods=["POST"])
def sync_company_action():
    """
    Deprecated: 同步公司行动数据
    """
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="COMPANY_ACTION_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for company action data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        def task_func():
            company_action_synchronizer.sync(progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating company action data sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/fundamental/sync", methods=["POST"])
def sync_fundamental():
    """
    同步公司基本面数据
    """
    data = request.get_json()
    stock_codes = data.get("stock_codes")
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="FUNDAMENTAL_DATA_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for fundamental data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        def task_func():
            fundamental_data_synchronizer.sync(stock_codes, progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating fundamental data sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/fundamental/sync_one_period", methods=["POST"])
def sync_fundamental_by_period():
    """
    同步公司基本面数据
    参数：
      - period: 财报周期，例如 "20250331" 表示2025年一季度
    """
    data = request.get_json()
    start_period = data.get("start_period")
    end_period = data.get("end_period")
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="FUNDAMENTAL_DATA_SYNC_ONE_PERIOD",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for one period fundamental data sync.", task_id)

        def task_func():
            fundamental_data_synchronizer.sync_by_period(start_period, end_period)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating fundamental data sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/suspend/sync_all", methods=["POST"])
def sync_suspend_all():
    """
    全量同步停复牌数据：使用 query 参数 date（格式如 "20220222"）
    """
    date_param = datetime.today().strftime("%Y%m%d") # 同步截止该日期的所有数据
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="SUSPEND_DATA_ALL_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for full suspend data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        def task_func():
            suspend_data_synchronizer.sync_all(date=date_param, progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating full suspend data sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/suspend/sync_today", methods=["POST"])
def sync_suspend_today():
    """
    增量同步当天的停复牌数据
    """
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="SUSPEND_DATA_TODAY_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for today suspend data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        def task_func():
            suspend_data_synchronizer.sync_today(progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating today suspend data sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/cninfo/share_change/sync", methods=["POST"])
def sync_cninfo_share_change():
    """
    Deprecated
    """
    try:
        new_task = TaskRecord(
            task_type="CNINFO_SHARE_CHANGE_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for stock info data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        # 定义后台任务函数
        def task_func():
            cninfo_stock_share_change_fetcher.fetch_cninfo_data(progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error cninfo share change data sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500
    

@fin_data_bp.route("/index_info/sync", methods=["POST"])
def sync_index_info():
    """
    同步指数基本信息数据
    """
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="INDEX_INFO_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for index info data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        # 定义后台任务函数
        def task_func():
            index_info_synchronizer.sync(progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating index info sync task")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/index_hist/sync", methods=["POST"])
def sync_index_hist():
    """
    创建一个同步股票历史数据的任务，并启动同步。
    请求参数（JSON）中可包含 task_type（例如 "INDEX_HIST_SYNC"），
    此处示例中直接固定为同步无复权数据。
    """
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="INDEX_HIST_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for index historical data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        def task_func():
            index_hist_synchronizer.sync(progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating index hist sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@fin_data_bp.route("/index_hist/sync_by_date", methods=["POST"])
def sync_index_hist_by_date():
    """
    同步指定交易日的指数历史行情数据。
    请求参数格式：
    {
        "start_date": "20240510",  # 格式必须为 YYYYMMDD
        "end_date": "20240510"  # 格式必须为 YYYYMMDD
        "target_code_list": ["000985.CSI", "000922.CSI"]
    }
    """
    try:
        req_data = request.get_json()
        start_date = req_data.get("start_date")
        end_date = req_data.get("end_date")
        target_code_list = req_data.get("target_code_list", [])

        if not start_date or not start_date.isdigit() or len(start_date) != 8 \
            or not end_date or not end_date.isdigit() or len(end_date) != 8:
            return jsonify({"status": "error", "message": "参数格式不正确，应为 'YYYYMMDD'"}), 400
        
        if not isinstance(target_code_list, list):
            return jsonify({"error": "target_code_list must be a list"}), 400

        # 创建任务记录
        new_task = TaskRecord(
            task_type="INDEX_HIST_SYNC_BY_DATE",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message=f"Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for index sync on %s to %s", task_id, start_date, end_date)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        def task_func():
            index_hist_synchronizer.sync_by_trade_date(start_date, end_date, target_code_list, progress_callback=progress_cb)

        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200

    except Exception as e:
        logger.exception("Error starting index sync by trade_date.")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@fin_data_bp.route("/fund_info/sync", methods=["POST"])
def sync_fund_info():
    """
    同步基金基本信息数据
    """
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="FUND_INFO_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for fund info data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        # 定义后台任务函数
        def task_func():
            fund_info_synchronizer.sync(progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating fund info sync task")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/fund_hist/sync", methods=["POST"])
def sync_fund_hist():
    """
    创建一个同步股票历史数据的任务，并启动同步。
    请求参数（JSON）中可包含 task_type（例如 "FUND_HIST_SYNC"），
    此处示例中直接固定为同步无复权数据。
    """
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="FUND_HIST_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for fund historical data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        def task_func():
            fund_hist_synchronizer.sync(progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating fund hist sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@fin_data_bp.route("/factors/sync_all", methods=["POST"])
def sync_factors_all():
    data = request.get_json()
    start = data.get("start_date")
    end = data.get("end_date")
    mode = data.get("mode")
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="FACTORS_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for factors data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        def task_func():
            factor_fetcher.fetch_all(start_date=start, end_date=end, mode=mode, progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating factors sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500
    
    
@fin_data_bp.route("/macro/fetch_all", methods=["POST"])
def fetch_all_macro_data():
    try:
        MacroDataFetcher.fetch_all()
        return jsonify({"status": "success", "message": "All macro data fetched and stored."}), 200
    except Exception as e:
        logger.exception("Error during fetch_all_macro_data")
        return jsonify({"status": "error", "message": str(e)}), 500


calender_fetcher = CalendarFetcher()
@fin_data_bp.route("/trade_cal", methods=["POST"])
def sync_trade_calendar():
    try:
        data = request.get_json()
        start = data.get("start_date")
        end = data.get("end_date")
        if not start or not end:
            return jsonify({"status": "error", "message": "start_date 和 end_date 必须提供"}), 400

        calender_fetcher.fetch_trade_calendar(start=start, end=end)
        return jsonify({"status": "success", "message": f"交易日历同步成功：{start} 到 {end}"}), 200
    except Exception as e:
        logger.exception("同步交易日失败")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/etf_info/sync", methods=["POST"])
def sync_etf_info():
    """
    同步所有etf信息至数据库
    """
    etf_data_fetcher = EtfDataFetcher()
    try:
        etf_data_fetcher.fetch_etf_info_all()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.exception("Error executing etf info sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/etf_hist/sync", methods=["POST"])
def sync_etf_hist_all():
    """
    同步所有etf历史数据至数据库
    """
    etf_data_fetcher = EtfDataFetcher()
    try:
        etf_data_fetcher.fetch_etf_hist_all()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.exception("Error executing etf hist sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/etf_hist/sync_by_date")
def sync_etf_hist_by_date():
    """
    同步指定时间段内的etf历史数据至数据库
    请求参数格式：
    {
        "start_date": "20240510",  # 格式必须为 YYYYMMDD
        "end_date": "20240510"  # 格式必须为 YYYYMMDD
    }
    """
    
    etf_data_fetcher = EtfDataFetcher()
    try:
        req_data = request.get_json()
        start_date = req_data.get("start_date")
        end_date = req_data.get("end_date")
        etf_data_fetcher.fetch_etf_hist_all_by_date(start_date, end_date)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.exception("Error executing etf hist sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@fin_data_bp.route("/update_all", methods=["POST"])
def update_all_fin_data():
    """
    更新所有财务数据
    Params:
        - start_date: str, 开始日期, 格式YYYYMMDD
        - end_date: str, 结束日期, 格式YYYYMMDD
    """
    try:
        data = request.get_json()
        start = data.get("start_date")
        end = data.get("end_date")
        mode = data.get("mode")
        if not start or not end:
            return jsonify({"status": "error", "message": "start_date 和 end_date 必须提供"}), 400
        if not mode:
            return jsonify({"status": "error", "message": "mode 必须提供 (realtime/history) "}), 400
        
        trade_dates = calender_fetcher.get_trade_date(start, end, format="%Y%m%d")
        
        stock_info_synchronizer.sync()

        for trade_date in trade_dates:
            stock_hist_synchronizer.sync_by_trade_date(trade_date)
            adj_factor_synchronizer.sync_by_trade_date(trade_date)

        suspend_data_synchronizer.sync_by_date(start_date=start, end_date=end)
        index_hist_synchronizer.sync_by_trade_date(start, end, ["000985.CSI", "000300.SH", "000001.SH", "399006.SZ", "000699.SH", "000905.SH", "000852.SH", "932000.CSI", "000922.CSI"])

        csi_index_fetcher = CSIIndexDataFetcher()
        csi_index_fetcher.fetch_and_store_csi_index_data(start_date=start, end_date=end)

        gold_data_fetcher = GoldDataFetcher()
        gold_data_fetcher.fetch_and_store_sge_index_data(start_date=start, end_date=end)

        fund_info_synchronizer.sync()
        fund_hist_synchronizer.sync_by_trade_date(start_date=start, end_date=end)

        etf_data_fetcher = EtfDataFetcher()
        etf_data_fetcher.fetch_etf_hist_all_by_date(start, end)

        start_date_fmt = pd.to_datetime(start, format="%Y%m%d").strftime("%Y-%m-%d")
        end_date_fmt = pd.to_datetime(end, format="%Y%m%d").strftime("%Y-%m-%d")

        factor_fetcher.fetch_all(start_date=start_date_fmt, 
                                 end_date=end_date_fmt,
                                 mode=mode)
        
        return jsonify({"message": "success"})
    except Exception as e:
        logger.exception(str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


from app.ml.beta_estimator import run_historical_beta_batch
from app.data.helper import get_all_fund_codes_with_source

@fin_data_bp.route("/dynamic_beta_hist", methods=["POST"])
def update_historical_dynamic_beta():
    data = request.get_json()
    fund_codes = data.get("fund_codes")
    asset_type = data.get("asset_type")
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    if not fund_codes:
        df_all = get_all_fund_codes_with_source(asset_type)
        type_codes_map = df_all.groupby("source")["fund_code"].apply(list).to_dict()

    if not start_date or not end_date:
        return jsonify({"status": "error", "message": "请提供 start_date 和 end_date"}), 400

    try:
        for asset_type, fund_codes in type_codes_map.items():
            run_historical_beta_batch(fund_codes, asset_type, start_date, end_date)
        return jsonify({
            "status": "success",
            "message": f"历史 Kalman β 已完成 {len(fund_codes)} 只基金的处理",
            "fund_count": len(fund_codes)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
from app.ml.beta_estimator import run_realtime_update_batch
from app.data_fetcher.calender_fetcher import CalendarFetcher
from app.data.helper import get_all_fund_codes_with_source

@fin_data_bp.route("/dynamic_beta", methods=["POST"])
def update_dynamic_beta():
    """
    fund_codes: list[str] 基金代码列表
    asset_type: str 资产类型， fund_info/etf_info
    start_date: str 开始日期 YYYY-MM-DD，若为空，则寻找上一次同步的日期后下一个交易日
    end_date: str 结束日期 YYYY-MM-DD，若为空，则使用当前日期的最近一个交易日
    """
    data = request.get_json()
    fund_codes = data.get("fund_codes")
    asset_type = data.get("asset_type")
    start_date = data.get("start_date", None)
    end_date = data.get("end_date")

    if not fund_codes:
        df_all = get_all_fund_codes_with_source(asset_type, end_date)
        fund_codes = df_all["fund_code"].tolist()

    if not end_date:
        latest_trade_date = CalendarFetcher().get_trade_date(start="19900101", end=pd.to_datetime("today").strftime("%Y%m%d"), format="%Y-%m-%d", limit=1, ascending=False)
        if not latest_trade_date:
            return jsonify({"status": "error", "message": "交易日历为空，请先同步"}), 400
        end_date = latest_trade_date[0]
    else:
        latest_trade_date = CalendarFetcher().get_trade_date(start="19900101", end=end_date.replace("-", ""), format="%Y-%m-%d", limit=1, ascending=False)
        if not latest_trade_date:
            return jsonify({"status": "error", "message": "交易日历为空，请先同步"}), 400
        end_date = latest_trade_date[0]

    try:
        run_realtime_update_batch(fund_codes, start_date=start_date, end_date=end_date)
        logging.info(f"[实时更新] 基金betas处理完成，处理{len(fund_codes)}个基金")
        return jsonify({
            "status": "success",
            "message": f"Kalman β 更新完成"
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
