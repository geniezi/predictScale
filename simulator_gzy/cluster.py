from node import Node
import threading
from multiprocessing import Process, Pool


class Cluster:
    def __init__(self):
        self.nodes = []
        self.tasks_queue = []
        self.waiting_tasks = []
        self.running_tasks = []
        self.task_index = 0

    def add_node(self, node):
        self.nodes.append(node)

    def add_task(self, task):
        self.tasks_queue.append(task)

    def update_started_tasks(self, current_time):
        while self.task_index < len(self.tasks_queue):
            task = self.tasks_queue[self.task_index]
            if task.start_time > current_time:
                break
            self.waiting_tasks.append(task)
            self.task_index += 1

    def time_step(self, current_time):
        # 单进程
        # 遍历全部节点，并为每个节点创建一个线程来检查任务是否已完成
        for node in self.nodes:
            self.check_and_free_resources_on_node( node, current_time)

        # 更新等待任务
        if self.task_index < len(self.tasks_queue):
            self.update_started_tasks(current_time)

    def check_and_free_resources_on_node(self, node, current_time):
        # 遍历全部节点，检查它们上面运行的任务是否已完成
        task_list = node.get_completed_tasks(current_time)
        # 遍历task_list，释放资源
        for task in task_list:
            print(task.task_id)
            self.running_tasks.remove(task)
            # 单进程
            node.free_resources(task)

    def assign_task(self, task, node, current_time):
        node.assign_task(task, current_time)
        self.waiting_tasks.remove(task)
        self.running_tasks.append(task)

    def get_first_assignable_node(self, task):
        for node in self.nodes:
            if node.check_if_assignable(task):
                return node
        return None

    def shutdown(self):
        pass
