import importlib
import math
import queue

import pandas as pd

time_sequence_length = 10


def get_response_time(requests, replicas):
    a = 300
    b = -0.00002632
    c = 0
    response_raw = a * math.exp(-b * requests / replicas) + c
    response = math.floor(response_raw)
    return response


def get_replica(requests, response):
    a = 300
    b = -0.00002632
    c = 0
    replicas = -(requests * (b / math.log((response - c) / a)))
    return (replicas)


def get_requests(replicas):
    a = 300
    b = -0.00002632
    c = 0
    response = 500
    requests = replicas / (-b) * math.log((response - c) / a)
    return math.floor(requests)


def get_cost(replicas):
    CORE_NUM = 2
    MEMORY = 2
    GPU = 1
    core_cost = 0.00003334
    memory_cost = 0.00001389
    gpu_cost = 0.00010002
    return (core_cost * CORE_NUM + memory_cost * MEMORY + gpu_cost * GPU) * replicas


def over_provision(data):
    data = data.copy()
    max_requests = data['requests'].max()
    max_replicas = math.ceil(get_replica(max_requests, 500))
    cost_by_second = get_cost(max_replicas)
    cost = cost_by_second * len(data)
    slo_requests, slo_time = 0, 0
    scale_time = 0
    return cost, slo_requests, slo_time, scale_time


def half_provision(data):
    data = data.copy()
    max_requests = data['requests'].max()
    half_replicas = math.ceil(get_replica(max_requests, 500) / 2)
    print(half_replicas)
    cost_by_second = get_cost(half_replicas)
    cost = cost_by_second * len(data)
    slo_requests, slo_time = 0, 0
    for i in range(len(data)):
        response = get_response_time(data['requests'][i], half_replicas)
        if response > 500:
            slo_requests += data['requests'][i] - get_requests(half_replicas)
            slo_time += 1

    scale_time = 0
    return cost, slo_requests / data['requests'].sum() * 100, slo_time / len(data) * 100, scale_time


def provision_95ile(data):
    data = data.copy()
    requests_95ile = data['requests'].quantile(0.95)
    replicas_95ile = math.ceil(get_replica(requests_95ile, 500))
    print(replicas_95ile)
    cost_by_second = get_cost(replicas_95ile)
    cost = cost_by_second * len(data)
    slo_requests, slo_time = 0, 0
    for i in range(len(data)):
        response = get_response_time(data['requests'][i], replicas_95ile)
        if response > 500:
            slo_requests += data['requests'][i] - get_requests(replicas_95ile)
            slo_time += 1
    scale_time = 0
    return cost, slo_requests / data['requests'].sum() * 100, slo_time / len(data) * 100, scale_time


def provision_99ile(data):
    data = data.copy()
    requests_99ile = data['requests'].quantile(0.99)
    replicas_99ile = math.ceil(get_replica(requests_99ile, 500))
    print(replicas_99ile)
    cost_by_second = get_cost(replicas_99ile)
    cost = cost_by_second * len(data)
    slo_requests, slo_time = 0, 0
    for i in range(len(data)):
        response = get_response_time(data['requests'][i], replicas_99ile)
        if response > 500:
            slo_requests += data['requests'][i] - get_requests(replicas_99ile)
            slo_time += 1
    scale_time = 0
    return cost, slo_requests / data['requests'].sum() * 100, slo_time / len(data) * 100, scale_time


def average_provision(data):
    data = data.copy()
    replicas = math.ceil(get_replica(data['requests'].mean(), 500))
    print(replicas)
    cost_by_second = get_cost(replicas)
    cost = cost_by_second * len(data)
    slo_requests, slo_time = 0, 0
    for i in range(len(data)):
        response = get_response_time(data['requests'][i], replicas)
        if response > 500:
            slo_requests += data['requests'][i] - get_requests(replicas)
            slo_time += 1
    scale_time = 0
    return cost, slo_requests / data['requests'].sum() * 100, slo_time / len(data) * 100, scale_time


def reactive(data):
    data = data.copy()
    replicas = 1
    cost = 0
    slo_requests, slo_time = 0, 0
    slo_satisfied = 0
    scale_time = 0
    for i in range(1, len(data)):
        response = get_response_time(data['requests'][i], replicas)
        cost_by_second = get_cost(replicas)
        cost += cost_by_second
        if response > 500:
            replicas += 1
            slo_requests += data['requests'][i]
            slo_time += 1
            slo_satisfied = 0
            scale_time += 1
        else:
            slo_satisfied += 1
        if slo_satisfied >= 5:
            if replicas > 1:
                replicas -= 1
                scale_time += 1
            slo_satisfied = 0
    return cost, slo_requests / data['requests'].sum() * 100, slo_time / len(data) * 100, scale_time


def hpa(data):
    # current_replica=last_minute_replica * response/500
    data = data.copy()
    replicas = 1
    cost = 0
    slo_requests, slo_time = 0, 0
    scale_time = 0
    for i in range(1, len(data)):
        response = get_response_time(data['requests'][i], replicas)
        cost_by_second = get_cost(replicas)
        cost += cost_by_second
        if response > 500:
            slo_requests += data['requests'][i] - get_requests(replicas)
            slo_time += 1
        replicas_new = math.ceil(replicas * response / 500)
        if replicas_new > 1:
            if replicas_new != replicas:
                scale_time += 1
                replicas = replicas_new
    return cost, slo_requests / data['requests'].sum() * 100, slo_time / len(data) * 100, scale_time


def my_strategy(data):
    data = data.copy()
    replicas = 1
    cost = 0
    slo_requests, slo_time = 0, 0
    scale_time = 0
    history_replicas = queue.Queue(10)
    last_burst=0
    for i in range(time_sequence_length):
        history_replicas.put(replicas)
        # 先使用hpa
        response = get_response_time(data['requests'][i], replicas)
        cost_by_second = get_cost(replicas)
        cost += cost_by_second
        if response > 500:
            slo_requests += data['requests'][i] - get_requests(replicas)
            slo_time += 1
        if i != time_sequence_length - 1:
            replicas_new = math.ceil(replicas * response / 500)
            if replicas_new > 1:
                if replicas_new != replicas:
                    scale_time += 1
                    replicas = replicas_new

    for i in range(time_sequence_length, len(data)):
        # predict = data['predict'][i]
        burst = data['burst'][i]
        if burst :
            # 有突发请求，使用历史记录中最大
            replicas_new = max(history_replicas.queue)
            if replicas_new != replicas:
                scale_time += 1
                replicas = replicas_new
            last_burst = 1
        else:
            # 没有突发请求，使用预测值
            replicas_new = data['replicas'][i]
            if replicas_new != replicas:
                scale_time += 1
                replicas = replicas_new
            last_burst = 0
        history_replicas.get()
        history_replicas.put(replicas)
        print(history_replicas.queue)
        response = get_response_time(data['requests'][i], replicas)
        cost_by_second = get_cost(replicas)
        cost += cost_by_second
        if response > 500:
            slo_requests += data['requests'][i] - get_requests(replicas)
            slo_time += 1
    return cost, slo_requests / data['requests'].sum() * 100, slo_time / len(data) * 100, scale_time


def my_strategy_without_burst(data):
    data = data.copy()
    replicas = 1
    cost = 0
    slo_requests, slo_time = 0, 0
    scale_time = 0
    for i in range(time_sequence_length):
        # 先使用hpa
        response = get_response_time(data['requests'][i], replicas)
        cost_by_second = get_cost(replicas)
        cost += cost_by_second
        if response > 500:
            slo_requests += data['requests'][i] - get_requests(replicas)
            slo_time += 1
        if i != time_sequence_length - 1:
            replicas_new = math.ceil(replicas * response / 500)
            if replicas_new > 1:
                if replicas_new != replicas:
                    scale_time += 1
                    replicas = replicas_new

    for i in range(time_sequence_length, len(data)):
        replicas_new = data['replicas'][i]
        if replicas_new != replicas:
            scale_time += 1
            replicas = replicas_new
        response = get_response_time(data['requests'][i], replicas)
        cost_by_second = get_cost(replicas)
        cost += cost_by_second
        if response > 500:
            slo_requests += data['requests'][i] - get_requests(replicas)
            slo_time += 1
    return cost, slo_requests / data['requests'].sum() * 100, slo_time / len(data) * 100, scale_time

if __name__ == "__main__":
    current_module = importlib.import_module('__main__')
    df = pd.read_csv('burst.csv')
    result = {
        'over_provision': [],
        'half_provision': [],
        'provision_95ile': [],
        'provision_99ile': [],
        'average_provision': [],
        'reactive': [],
        'hpa': [],
        'my_strategy': [],
        'my_strategy_without_burst': [],
    }
    for def_name in result.keys():
        result[def_name] = getattr(current_module, def_name)(df)

    # 保存到csv，列名为stragety，cost, slo_violation, scale_time

    data_to_save = []

    # 循环result字典，将每个策略的结果追加到待保存列表中
    for strategy_name, values in result.items():
        data_row = [strategy_name] + list(values)  # 把策略名称和返回值组合成一行
        data_to_save.append(data_row)

    # 构造DataFrame，指定列名
    df_to_save = pd.DataFrame(data_to_save, columns=['stragety', 'cost', 'slo_requests', 'slo_time', 'scale_time'])

    # 保存DataFrame到CSV文件
    df_to_save.to_csv('scale.csv', index=False)  # index=False表示不保存行索引到CSV文件
