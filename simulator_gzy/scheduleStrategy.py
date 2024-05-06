class ScheduleStrategy:
    def __init__(self, cluster):
        self.cluster = cluster

    def schedule(self, current_time):
        # 这里将任务调度到节点，由子类具体实现
        raise NotImplementedError


class FirstFitScheduleStrategy(ScheduleStrategy):
    def schedule(self, current_time):
        index = 0
        while len(self.cluster.waiting_tasks) > 0 and index < len(self.cluster.nodes):
            task = self.cluster.waiting_tasks[0]
            node = self.cluster.get_first_assignable_node(task)
            if node:
                self.cluster.assign_task(task, node, current_time)
            else:
                index += 1


class LastFitScheduleStrategy(ScheduleStrategy):
    def schedule(self, current_time):
        index = 0
        while len(self.cluster.waiting_tasks) > 0 and index < len(self.cluster.nodes):
            task = self.cluster.waiting_tasks[0]
            node = self.cluster.get_last_assignable_node(task)
            if node:
                self.cluster.assign_task(task, node, current_time)
            else:
                index += 1
