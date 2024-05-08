import time
from datetime import datetime

import config
import utils
import argparse
from cluster import Cluster
import argparse
import time
from datetime import datetime

import config
import utils
from cluster import Cluster


class Simulator:
    def __init__(self, scheduler_config, start_date, end_date, num_nodes_mul, save_path):
        self.cluster = Cluster()
        self.schedulerConfig = scheduler_config  # 调度策略
        self.scheduler = config.scheduleStrategyConfig[self.schedulerConfig](self.cluster)
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.start_time = self.start_date.timestamp()
        self.end_time = self.end_date.timestamp()
        self.current_time = self.start_time
        self.save_path=save_path

        self.node_list = utils.init_node_list(num_nodes_mul)
        for node in self.node_list:
            self.cluster.add_node(node)
        self.task_list = utils.init_task_list()
        for task in self.task_list:
            self.cluster.add_task(task)
        print(f"Simulator initialization completed.")

    def run(self):
        start = time.time()
        while self.cluster.task_index < len(self.cluster.tasks_queue):
            self.current_time += 1
            self.cluster.time_step(self.current_time)  # 模拟时间流逝
            self.scheduler.schedule(self.current_time)  # 调度任务
            print(
                f"Current time: {self.current_time}/{self.end_time}, Running tasks: {len(self.cluster.running_tasks)}, Waiting tasks: {len(self.cluster.waiting_tasks)}")
            # 限制时间，用于调试
            if self.current_time >= self.end_time:
                break
        end = time.time()
        print("Total time cost: ", end - start)
        if self.current_time >= self.end_time:
            return
        while len(self.cluster.running_tasks) > 0:
            self.current_time += 1
            self.cluster.time_step(self.current_time)
            print(f"Current time: {self.current_time}/{self.end_time}")

    def save_results(self):
        # 保存任务的性能数据
        performance_data = []
        for task in self.cluster.assigned_task:
            performance_data.append(task.get_performance_data(self.current_time))
        utils.save_performance_data(performance_data, self)

    def shutdown(self):
        self.cluster.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator.')
    parser.add_argument('--scheduler', default='FirstFit', type=str, help='Scheduler Policy')
    parser.add_argument('-sd', '--start_date', default='1970-01-13', type=str, help='Start Date')
    parser.add_argument('-ed', '--end_date', default='1970-01-15', type=str, help='End Date')
    parser.add_argument('-nm', '--num_nodes_mul', default=40, type=int, help='Num of Nodes Multiplier')
    parser.add_argument('-sp', '--save_path', default='simulator_gzy/result/', type=str, help='Save Path')

    args = parser.parse_args()
    SCHEDULER = args.scheduler
    START_DATE = args.start_date
    END_DATE = args.end_date
    NUM_NODES_MUL = args.num_nodes_mul
    SAVE_PATH = args.save_path

    # 获取命令行参数
    simulator = Simulator(
        scheduler_config=SCHEDULER,
        start_date=START_DATE,
        end_date=END_DATE,
        num_nodes_mul=NUM_NODES_MUL,
        save_path=SAVE_PATH)
    simulator.run()
    simulator.save_results()
    simulator.shutdown()
