# app/routes/task_routes.py
from flask import Blueprint, request, jsonify
import logging
import datetime
from app.dao.task_record_dao import task_record_dao
from app.data.fetcher import stock_hist_synchronizer

logger = logging.getLogger(__name__)
task_bp = Blueprint("task", __name__, url_prefix="/tasks")

# 单例 TaskRecordDao
task_dao = task_record_dao

@task_bp.route("/<int:task_id>", methods=["GET"])
def query_task_progress(task_id: int):
    """
    查询指定任务的进度信息，返回任务记录的相关字段。
    """
    task = task_dao.get_by_id(task_id)
    if task:
        return jsonify({
            "task_id": task.id,
            "task_type": task.task_type,
            "task_status": task.task_status,
            "create_time": task.create_time.isoformat() if task.create_time else None,
            "update_time": task.update_time.isoformat() if task.update_time else None,
            "progress_current": task.progress_current,
            "progress_total": task.progress_total,
            "message": task.message
        }), 200
    else:
        return jsonify({"status": "error", "message": f"Task id {task_id} not found."}), 404
