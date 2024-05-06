import csv
import sys

import config
from node import Node
from task import Task

sys.path.append("util")
import osUtil as OSUtil

# import util.osUtil as OSUtil

# CSV 文件路径
csv_file = 'data_random/merge_dft.csv'
result_path = 'simulator_gzy/result/'


def init_node_list():
    node_list = []
    node_id = 0
    mul = 50  # 控制机器数量
    for _ in range(10 * mul):  # low-ended machines first
        node_list.append(Node(node_id, 96, 512, 2, gpu_type='T4'))
        node_id += 1
    for _ in range(13 * mul):  # low-ended machines first
        node_list.append(Node(node_id, 64, 512, 2, gpu_type='P100'))
        node_id += 1
    for _ in range(5 * mul):
        node_list.append(Node(node_id, 96, 512, 8, gpu_type='V100'))
        node_id += 1
    print("节点数量：", len(node_list))
    return node_list


def init_task_list():
    task_list = []
    task_id = 0
    # 读取CSV文件并创建任务实例
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # num = int(float(row['inst_num']))
            for num in range(int(float(row['inst_num']))):
                task = Task(
                    task_id=task_id,
                    task_name=row['task_name'],
                    start_time=int(float(row['start_time'])),
                    run_time=int(float(row['run_time'])),
                    plan_cpu=float(row['plan_cpu']),
                    plan_mem=float(row['plan_mem']),
                    plan_gpu=float(row['plan_gpu']) if row['plan_gpu'] != '' else 0,
                    gpu_type=row['gpu_type'],
                    task_type=row['task_type']
                )
                task_list.append(task)
                task_id += 1
    print("任务数量：", len(task_list))
    return task_list


def save_performance_data(performance_data, simulator):
    # 保存任务性能数据到CSV文件
    # 文件名格式：调度策略_开始日期_结束日期_节点数_任务数
    result_name = simulator.schedulerConfig + '_' + config.start_date.strftime(
        '%Y-%m-%d') + '_' + config.end_date.strftime('%Y-%m-%d') + '_' + str(
        list.__sizeof__(simulator.node_list)) + 'nodes_' + str(list.__sizeof__(simulator.task_list)) + 'tasks'
    save_dir = OSUtil.create_folder_with_incrementing_number(result_path, result_name)

    with open(save_dir + 'performance_data.csv', 'w', newline='') as file:
        fieldnames = config.SAVE_FIELDS
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for data in performance_data:
            writer.writerow(data)
    print("Performance data saved to performance_data.csv")
