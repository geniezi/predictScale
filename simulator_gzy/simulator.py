import time

from cluster import Cluster
import utils
import config
from node import Node
from task import Task
import datetime


date_to_convert = datetime.datetime(1970, 1, 15)
end_time = int(time.mktime(date_to_convert.timetuple()))  # 调度任务截止时间


class Simulator:
    def __init__(self):
        self.cluster = Cluster()
        self.schedulerConfig = "FirstFit"  # 调度策略
        self.scheduler = config.scheduleStrategyConfig[self.schedulerConfig](self.cluster)
        self.node_list = utils.init_node_list()
        for node in self.node_list:
            self.cluster.add_node(node)
        self.task_list = utils.init_task_list()
        for task in self.task_list:
            self.cluster.add_task(task)
        self.current_time = 1036804
        print(f"Simulator initialization completed.")

    def run(self):

        while self.cluster.task_index < len(self.cluster.tasks_queue):
            self.current_time += 1
            print(f"Current time: {self.current_time}/{end_time}")
            self.cluster.time_step(self.current_time)  # 模拟时间流逝
            self.scheduler.schedule(self.current_time)  # 调度任务
            if self.current_time== end_time:
                break
        if self.current_time== end_time:
            return
        while self.cluster.running_tasks.__sizeof__() > 0:
            self.current_time += 1
            print(f"Current time: {self.current_time}/{end_time}")
            self.cluster.time_step(self.current_time)

    def save_results(self):
        # 保存任务的性能数据
        performance_data = []
        for task in self.cluster.assigned_task:
            performance_data.append(task.get_performance_data())
        utils.save_performance_data(performance_data, self.schedulerConfig)

    def shutdown(self):
        self.cluster.shutdown()


if __name__ == '__main__':
    simulator = Simulator()
    simulator.run()
    simulator.save_results()
    simulator.shutdown()
