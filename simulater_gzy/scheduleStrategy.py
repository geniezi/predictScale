from cluster import Cluster
from node import Node


class ScheduleStrategy:
    def __init__(self, cluster):
        self.cluster = cluster

    def schedule(self, current_time):
        # 这里将任务调度到节点，由子类具体实现
        raise NotImplementedError


class FirstFitScheduleStrategy(ScheduleStrategy):
    def schedule(self, current_time):
        for task in self.cluster.waiting_tasks:
            node = self.cluster.get_first_assignable_node(task)
            if node:
                self.cluster.assign_task(task, node, current_time)
            else:
                print(f"No available node to run Task {task.task_name} (id: {task.task_id})")
