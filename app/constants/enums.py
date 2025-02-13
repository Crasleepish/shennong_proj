from enum import Enum

class TaskType(Enum):
    UPDATE_ADJ = (1, '更新前复权数据')

    def __init__(self, value, description):
        self.description = description

class TaskStatus(Enum):
    INIT = (1, '已创建待执行')
    RUNNING = (2, '正在执行')
    DONE = (3, '已完成')

    def __init__(self, value, description):
        self.description = description

        