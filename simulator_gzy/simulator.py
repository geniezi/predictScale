import sys
import time

import config
import utils
from cluster import Cluster

start_time = config.start_time
end_time = config.end_time


class Simulator:
    def __init__(self, schedulerConfig):
        self.cluster = Cluster()
        self.schedulerConfig = schedulerConfig  # 调度策略
        self.scheduler = config.scheduleStrategyConfig[self.schedulerConfig](self.cluster)
        self.node_list = utils.init_node_list()
        for node in self.node_list:
            self.cluster.add_node(node)
        self.task_list = utils.init_task_list()
        for task in self.task_list:
            self.cluster.add_task(task)
        self.current_time = start_time
        print(f"Simulator initialization completed.")

    def run(self):
        start = time.time()
        while self.cluster.task_index < len(self.cluster.tasks_queue):
            self.current_time += 1
            self.cluster.time_step(self.current_time)  # 模拟时间流逝
            self.scheduler.schedule(self.current_time)  # 调度任务
            print(f"Current time: {self.current_time}/{end_time}")
            # 限制时间，用于调试
            if self.current_time >= end_time:
                break
        end = time.time()
        print("Total time cost: ", end - start)
        if self.current_time >= end_time:
            return
        while len(self.cluster.running_tasks) > 0:
            self.current_time += 1
            self.cluster.time_step(self.current_time)
            print(f"Current time: {self.current_time}/{end_time}")

    def save_results(self):
        # 保存任务的性能数据
        performance_data = []
        for task in self.cluster.assigned_task:
            performance_data.append(task.get_performance_data(self.current_time))
        utils.save_performance_data(performance_data, self)

    def shutdown(self):
        self.cluster.shutdown()


if __name__ == '__main__':
    # 获取命令行参数
    if len(sys.argv) >= 2:
        scheduler = sys.argv[1]
    else:
        scheduler = "FirstFit"
    simulator = Simulator(scheduler)
    simulator.run()
    simulator.save_results()
    simulator.shutdown()
