import math

import numpy as np

from model.data.workload.dataInitial import get_workload, dataLength, trainLength, \
    servicesDict, costDict, sloViolateDict, apiLatencyData, microservicesPodData, latency_latency
from base_model import Cost

# HPA反应式弹性伸缩
# 获取工作负载
basePath, filePath, workloads = get_workload()
latencyDistribute = []
podDistribute = []
for minute in range(0, dataLength - trainLength):
    print(f"minute: {minute}")
    print(f"workload: {workloads[minute]}")
    # 获取工作负载
    for serviceName in servicesDict.keys():
        servicesDict[serviceName].workloadsDict["Total"] = workloads[minute]
    for serviceName in servicesDict.keys():
        print(f"service: {serviceName}")
        print(f"service latency Up: {servicesDict[serviceName].slo[0]}")
        print(f"service latency Down: {servicesDict[serviceName].slo[1]}")
        print(f"latency now: {servicesDict[serviceName].compute_basic_latency()}")
        print(f"service pod: {servicesDict[serviceName].pod}")
        print("-----------------------------------------------------")
    tmpData = []
    for serviceName in servicesDict.keys():
        costDict[serviceName]["cpu"] += Cost.get_cpu_cost() * servicesDict[serviceName].resourceRequirement["cpu"] \
                                        * servicesDict[serviceName].pod
        costDict[serviceName]["memory"] += \
            Cost.get_memory_cost() * servicesDict[serviceName].resourceRequirement["memory"] \
            * servicesDict[serviceName].pod
        costDict[serviceName]["gpu"] += Cost.get_gpu_cost() * servicesDict[serviceName].resourceRequirement["gpu"] \
                                        * servicesDict[serviceName].pod
        tmpData.append(servicesDict[serviceName].pod)
        podDistribute.append(servicesDict[serviceName].pod)
    # 遍历每个微服务
    for serviceName in servicesDict.keys():
        latencyDistribute.append(servicesDict[serviceName].compute_basic_latency())
        if servicesDict[serviceName].compute_basic_latency() > servicesDict[serviceName].slo[0]:
            servicesDict[serviceName].violate += 1
            if servicesDict[serviceName].pod <= servicesDict[serviceName].podThreshold[1]:
                servicesDict[serviceName].noViolate = 0
                servicesDict[serviceName].pod = servicesDict[serviceName].pod * math.ceil(servicesDict[serviceName].compute_basic_latency() / servicesDict[serviceName].slo[0])
                servicesDict[serviceName].operations += 1

        else:
            servicesDict[serviceName].noViolate += 1
            if servicesDict[serviceName].noViolate >= servicesDict[serviceName].keepScaleThreshold:
                if servicesDict[serviceName].pod > servicesDict[serviceName].podThreshold[0]:
                    servicesDict[serviceName].pod -= 1
                    servicesDict[serviceName].noViolate = 0
                    servicesDict[serviceName].operations += 1
    microservicesPodData.append(tmpData)

print("HPA: ")

sloViolateDataList = []
costDataList = []
operationsCostDataList = []
for requestType in sloViolateDict.keys():
    print(f"{requestType} violate SLO times: {sloViolateDict[requestType]}")
    sloViolateDataList.append(sloViolateDict[requestType])
for serviceName in servicesDict.keys():
    print(f"{serviceName} violate SLO times: {servicesDict[serviceName].violate}")
for serviceName in servicesDict.keys():
    print(f"{serviceName} scale operations: {servicesDict[serviceName].operations}")

for serviceName in servicesDict.keys():
    print(f"cpu cost: {costDict[serviceName]['cpu']}")
    print(f"memory cost: {costDict[serviceName]['memory']}")
    print(f"gpu cost: {costDict[serviceName]['gpu']}")
    costDataList.append(costDict[serviceName]['cpu'])
    costDataList.append(costDict[serviceName]['memory'])
    costDataList.append(costDict[serviceName]['gpu'])
    print(f"cost: {costDict[serviceName]['cpu'] + costDict[serviceName]['memory'] + costDict[serviceName]['gpu']}")
    operationsCostDataList.append(servicesDict[serviceName].operations)

    # # 指定要保存到的文件路径
    # file_path = "hpa/hpa_world_cup_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    #
    # file_path = "hpa/hpa_world_cup_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')
    #
    # print("内容已保存到文件:", file_path)

    # 指定要保存到的文件路径
    # file_path = "hpa/hpa_wiki_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    #
    # file_path = "hpa/hpa_wiki_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')
    #
    # print("内容已保存到文件:", file_path)

    # file_path = "hpa/hpa_s1_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    #
    # file_path = "hpa/hpa_s1_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')
    #
    # print("内容已保存到文件:", file_path)

    file_path = "hpa/hpa_s2_latency.txt"
    # 将整数数组保存到文本文件
    np.savetxt(file_path, latencyDistribute, fmt='%d')

    file_path = "hpa/hpa_s2_pod.txt"
    # 将整数数组保存到文本文件
    np.savetxt(file_path, podDistribute, fmt='%d')

    print("内容已保存到文件:", file_path)



# # 算法名称
# methodName = "HPA"
# # 数据集名称
# dataSetName = "worldCup"
# save_experiment_result_of_slo_violate_record(sloViolateDataList, f"sloViolateRecord/{methodName}/{dataSetName}_SLO_VIOLATE.xlsx")
# save_experiment_result_of_cost_record(costDataList, f"costRecord/{methodName}/{dataSetName}_COST.xlsx")
# save_experiment_result_of_api_latency(apiLatencyData, f"latencyRecord/{methodName}/{dataSetName}_LATENCY.xlsx")
# save_experiment_result_of_pod_scale(microservicesPodData, f"podRecord/{methodName}/{dataSetName}_POD.xlsx")
# save_experiment_result_of_scale_operations(microservicesPodData, f"podRecord/{methodName}/{dataSetName}_POD.xlsx")
