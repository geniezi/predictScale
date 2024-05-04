from cluster import Cluster
import utils
from node import Node
import scheduleStrategy
from task import Task

scheduleStrategyConfig = {
    "FirstFit": scheduleStrategy.FirstFitScheduleStrategy,
    # "BestFit": scheduleStrategy.BestFitScheduleStrategy,
    # "WorstFit": scheduleStrategy.WorstFitScheduleStrategy,
}


class Simulator:
    def __init__(self):
        self.cluster = Cluster()
        self.schedulerConfig = "FirstFit"  # 调度策略
        self.scheduler = scheduleStrategyConfig[self.schedulerConfig](self.cluster)
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
            print(f"Current time: {self.current_time}")
            self.cluster.time_step(self.current_time)  # 模拟时间流逝
            self.scheduler.schedule(self.current_time)  # 调度任务
        while self.cluster.running_tasks.__sizeof__() > 0:
            self.current_time += 1
            print(f"Current time: {self.current_time}")
            self.cluster.time_step(self.current_time)

    def save_results(self):
        # 保存任务的性能数据
        performance_data = []
        for task in self.task_list:
            performance_data.append(task.get_performance_data())
        utils.save_performance_data(performance_data, self.schedulerConfig)


if __name__ == '__main__':
    simulator = Simulator()
    simulator.run()
    simulator.save_results()
