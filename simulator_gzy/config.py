import datetime
import time

import scheduleStrategy

gpuConfig = {
    "T4": 1,
    "P100": 2,
    "V100": 3
}

scheduleStrategyConfig = {
    "FirstFit": scheduleStrategy.FirstFitScheduleStrategy,  # 节点按配置从低到高排序，因此先分配给配置最低的节点
    "LastFit": scheduleStrategy.LastFitScheduleStrategy,  # 先分配给配置最高的节点
    # "BestFit": scheduleStrategy.BestFitScheduleStrategy,
    # "WorstFit": scheduleStrategy.WorstFitScheduleStrategy,
}

SAVE_FIELDS = ['task_id',
               'start_time',
               'require_time',
               'start_run_time',
               'end_run_time',
               'waiting_time',
               'prefix_time',
               'speedup',
               'node_id',]

start_date = datetime.datetime(1970, 1, 13)
start_time = int(time.mktime(start_date.timetuple()))  # 调度任务开始时间
end_date = datetime.datetime(1970, 1, 14)
end_time = int(time.mktime(end_date.timetuple()))  # 调度任务截止时间
