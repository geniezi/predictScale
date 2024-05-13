import math

import pandas as pd


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


def get_cost(replicas):
    CORE_NUM = 2
    MEMORY = 2
    GPU = 1
    core_cost = 0.00003334
    memory_cost = 0.00001389
    gpu_cost = 0.00010002
    return (core_cost * CORE_NUM + memory_cost * MEMORY + gpu_cost * GPU) * replicas


def new_over_provision(data):
    data = data.copy()
    max_requests = data['requests'].max()
    max_replicas = math.ceil(get_replica(max_requests, 500))
    cost_by_second = get_cost(max_replicas)
    cost = cost_by_second * len(data)
    slo_violation = 0
    scale_time = 0
    return cost, slo_violation, scale_time

def half_provision(data):
    data = data.copy()
    max_requests = data['requests'].max()
    half_replicas = get_replica(max_requests, 500)/2
    cost_by_second = get_cost(half_replicas)
    cost = cost_by_second * len(data)
    slo_violation=0
    for i in range(len(data)):
        response = get_response_time(data['requests'][i], half_replicas)
        if response > 500:
            slo_violation += data['requests'][i]
    scale_time = 0
    return cost, slo_violation/data['requests'].sum()*100, scale_time

df = pd.read_csv('burst.csv')
result = {
    'over_provision': [],
    'half_provision': [],
}

result['over_provision'] = new_over_provision(df)
result['half_provision'] = half_provision(df)

print(result)