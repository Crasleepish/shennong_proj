# fin_data_routes.py
from flask import Blueprint, request, jsonify
import logging

# 导入各同步器（注意根据实际项目的模块路径调整）
from app.data.fetcher import stock_info_synchronizer, stock_hist_synchronizer, stock_adj_hist_synchronizer, company_action_synchronizer, fundamental_data_synchronizer, suspend_data_synchronizer
from app.data.index_fetcher import index_info_synchronizer, index_hist_synchronizer
from app.data.cninfo_fetcher import cninfo_stock_share_change_fetcher
from app.dao.task_record_dao import task_record_dao
from app.utils.async_task import launch_background_task
from app.models.task_record import TaskRecord


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
            stock_hist_synchronizer.sync(progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating stock hist sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/stock_adj_hist/sync", methods=["POST"])
def sync_stock_hist_adj():
    """
    同步股票前复权历史行情数据
    """
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        new_task = TaskRecord(
            task_type="STOCK_ADJ_HIST_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for stock adjusted historical data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        def task_func():
            stock_adj_hist_synchronizer.sync(progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200

    except Exception as e:
        logger.exception("Error creating forward-adjusted stock historical data sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/company_action/sync", methods=["POST"])
def sync_company_action():
    """
    同步公司行动数据
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
            fundamental_data_synchronizer.sync(progress_callback=progress_cb)

        # 启动后台任务
        launch_background_task(task_id, task_func)

        return jsonify({"status": "success", "task_id": task_id, "message": "Task started"}), 200
    except Exception as e:
        logger.exception("Error creating fundamental data sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500

@fin_data_bp.route("/suspend/sync_all", methods=["POST"])
def sync_suspend_all():
    """
    全量同步停复牌数据：使用 query 参数 date（格式如 "20120222"）
    """
    date_param = "20120222" # 实际停复牌数据从2012-2-22开始，同步该日期及之后的所有数据
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
    
@fin_data_bp.route("/all_hist/sync", methods=["POST"])
def sync_all_hist():
    """
    同步行情（复权/不复权）相关全部数据
    """
    try:
        # 创建任务记录（初始状态为 RUNNING，进度为 0）
        from app.models.task_record import TaskRecord
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

        # 调用同步器，并传入进度回调
        stock_info_synchronizer.sync(progress_callback=progress_cb)

        # 同步完成后，将任务状态更新为 DONE
        task_dao.update_status(task_id, "DONE", "Task completed.")
		
		##################################################
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
        logger.info("Created task id %d for stock history data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        # 调用同步器，并传入进度回调
        stock_hist_synchronizer.sync(progress_callback=progress_cb)

        # 同步完成后，将任务状态更新为 DONE
        task_dao.update_status(task_id, "DONE", "Task completed.")
		
		##################################################
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

        # 调用同步器，并传入进度回调
        company_action_synchronizer.sync(progress_callback=progress_cb)

        # 同步完成后，将任务状态更新为 DONE
        task_dao.update_status(task_id, "DONE", "Task completed.")
		
		##################################################
		# 创建任务记录（初始状态为 RUNNING，进度为 0）

        new_task = TaskRecord(
            task_type="STOCK_ADJ_HIST_SYNC",
            task_status="RUNNING",
            progress_current=0,
            progress_total=0,
            message="Task started."
        )
        new_task = task_dao.insert(new_task)
        task_id = new_task.id
        logger.info("Created task id %d for stock adjusted history data sync.", task_id)

        # 定义进度回调函数
        progress_cb = make_progress_callback(task_id)

        # 调用同步器，并传入进度回调
        stock_adj_hist_synchronizer.sync(progress_callback=progress_cb)

        # 同步完成后，将任务状态更新为 DONE
        task_dao.update_status(task_id, "DONE", "Task completed.")
		
		
        return jsonify({"status": "success", "task_id": task_id, "message": "Stock historical all data sync completed."}), 200
    except Exception as e:
        logger.exception("Error executing stock hist sync task.")
        return jsonify({"status": "error", "message": str(e)}), 500
    

@fin_data_bp.route("/cninfo/share_change/sync", methods=["POST"])
def sync_cninfo_share_change():
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
    同步股票基本信息数据
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