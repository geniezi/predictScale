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
               'node_id',
               'status']

