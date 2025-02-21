import threading
import logging
from app.dao.task_record_dao import TaskRecordDao

logger = logging.getLogger(__name__)

def launch_background_task(task_id: int, task_func, *args, **kwargs):
    """
    启动一个后台线程执行任务函数，并在任务执行结束后更新任务状态。
    
    参数:
      task_id: 任务记录的 ID
      task_func: 要执行的任务函数
      *args, **kwargs: 传递给任务函数的参数
    """
    task_dao = TaskRecordDao._instance

    def background():
        try:
            task_func(*args, **kwargs)
            task_dao.update_status(task_id, "DONE", "Task completed.")
            logger.info("Task id %d completed successfully.", task_id)
        except Exception as e:
            logger.exception("Error executing background task for task id %d", task_id)
            task_dao.update_status(task_id, "FAILED", f"Task failed: {str(e)}")
    
    t = threading.Thread(target=background)
    t.start()
