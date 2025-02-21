from sqlalchemy import Column, Integer, String, DateTime, Float
from app.database import Base
from datetime import datetime

class TaskRecord(Base):
    __tablename__ = "task_record"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_type = Column(String(50), nullable=False)       # 任务类型，例如 "STOCK_HIST_SYNC"
    task_status = Column(String(20), nullable=False)       # 任务状态，如 INIT, RUNNING, DONE, ERROR
    create_time = Column(DateTime, nullable=False, default=datetime.now)
    update_time = Column(DateTime, nullable=True)
    progress_current = Column(Float, nullable=True)        # 当前进度（如处理记录数）
    progress_total = Column(Float, nullable=True)          # 总计进度（如总记录数）
    message = Column(String(200), nullable=True)           # 可选：进度说明或错误信息

    def __repr__(self):
        return f"<TaskRecord(id={self.id}, type={self.task_type}, status={self.task_status})>"
