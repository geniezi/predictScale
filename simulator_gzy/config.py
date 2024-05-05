import scheduleStrategy

gpuConfig = {
    "T4": 1,
    "P100": 2,
    "V100": 3
}

scheduleStrategyConfig = {
    "FirstFit": scheduleStrategy.FirstFitScheduleStrategy,
    # "BestFit": scheduleStrategy.BestFitScheduleStrategy,
    # "WorstFit": scheduleStrategy.WorstFitScheduleStrategy,
}