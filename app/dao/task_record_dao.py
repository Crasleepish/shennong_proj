# app/dao/task_record_dao.py
from app.models.task_record import TaskRecord
from app.database import get_db
import logging
import datetime

logger = logging.getLogger(__name__)

class TaskRecordDao:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TaskRecordDao, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def insert(self, task: TaskRecord) -> TaskRecord:
        with get_db() as db:
            db.add(task)
            db.commit()
            db.refresh(task)
            return task

    def update_progress(self, task_id: int, current: float, total: float, message: str = None):
        with get_db() as db:
            task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
            if task:
                task.progress_current = current
                task.progress_total = total
                task.message = message
                task.update_time = datetime.datetime.now()
                db.commit()
            else:
                logger.error("Task id %s not found for progress update", task_id)

    def update_status(self, task_id: int, status: str, message: str = None):
        with get_db() as db:
            task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
            if task:
                task.task_status = status
                task.message = message
                task.update_time = datetime.datetime.now()
                db.commit()
            else:
                logger.error("Task id %s not found for status update", task_id)

    def get_by_id(self, task_id: int) -> TaskRecord:
        with get_db() as db:
            return db.query(TaskRecord).filter(TaskRecord.id == task_id).first()

task_record_dao = TaskRecordDao()