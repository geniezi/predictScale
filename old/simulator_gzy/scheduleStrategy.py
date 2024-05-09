class ScheduleStrategy:
    def __init__(self, cluster):
        self.cluster = cluster

    def schedule(self, current_time):
        # 这里将任务调度到节点，由子类具体实现
        raise NotImplementedError


class FirstFitScheduleStrategy(ScheduleStrategy):
    def schedule(self, current_time):
        temp_task = []
        for task in self.cluster.waiting_tasks:
            node = self.cluster.get_first_assignable_node(task)
            if node:
                self.cluster.assign_task(task, node, current_time)
            else:
                temp_task.append(task)
        self.cluster.waiting_tasks = temp_task


class LastFitScheduleStrategy(ScheduleStrategy):
    def schedule(self, current_time):
        temp_task = []
        for task in self.cluster.waiting_tasks:
            node = self.cluster.get_last_assignable_node(task)
            if node:
                self.cluster.assign_task(task, node, current_time)
            else:
                temp_task.append(task)
        self.cluster.waiting_tasks = temp_task
