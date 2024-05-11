import math

import numpy as np
from base_model import Cost
from burstDetect import burst_detect, get_max_pod
from model.data.workload.dataInitial import get_workload, dataLength, trainLength, \
    servicesDict, costDict, sloViolateDict, apiLatencyData, get_total_workload, residualLength, \
    inputLength
from model.resource_estimate_model import get_resource_estimators
from model.workload_predict_model import get_workload_predictors, get_scaler_and_train_data, split_data

basePath, filePath, workloads = get_workload()
basePath, filePath, totalWorkloads = get_total_workload()
# 工作负载预测器集合
workloadPredictors = {}
# scaler集合
workloadScalerDict = {}
# 工作负载历史记录集合
historyWorkloadDict = {}
violateTimes = 0
for serviceName in servicesDict.keys():
    # scaler和历史工作负载
    workloadScalerDict[serviceName], historyWorkloadDict[serviceName] = \
        get_scaler_and_train_data(totalWorkloads)
    # 工作负载预测器 基于融合GRU模型和残差分析的工作负载预测模型
    workloadPredictors[serviceName] = get_workload_predictors(basePath, filePath, historyWorkloadDict[serviceName], serviceName)
# 资源估计器集合
resourceEstimatorsDict = {}
for serviceName in servicesDict.keys():
    resourceEstimatorsDict[serviceName] = get_resource_estimators(serviceName, basePath, filePath)

for minute in range(trainLength - residualLength, trainLength):
    for serviceName in servicesDict.keys():
        servicesDict[serviceName].add_workload_history(totalWorkloads[minute])

k = 10
kw = 10
# 收集残差
for minute in range(trainLength - residualLength, trainLength):
    for serviceName in servicesDict.keys():
        tmp_data = workloadScalerDict[serviceName].transform(totalWorkloads[minute - inputLength: minute].reshape(-1, 1)).reshape(-1)
        input_data = [split_data(tmp_data)]
        predictedWorkload = int(workloadScalerDict[serviceName].inverse_transform(
            workloadPredictors[serviceName].predict(np.array(input_data)))[0][0])
        print(f"minute: {minute}")
        print(f"predictWorkload: {predictedWorkload}")
        print(f"actual workload: {totalWorkloads[minute]}")
        servicesDict[serviceName].workloadsDict["Total"] = predictedWorkload
        servicesDict[serviceName].add_workload_predict_error_history(totalWorkloads[minute])
        servicesDict[serviceName].add_workload_imbalance(totalWorkloads[minute: minute + 1] -
                                                         totalWorkloads[minute - 1: minute])
        if minute - trainLength - residualLength >= k:
            slope = (totalWorkloads[minute] - totalWorkloads[minute - k]) / k
            servicesDict[serviceName].slopeList.append(slope)
map = {}
latencyDistribute = []
podDistribute = []
predictWorkloadDistribute = []
workloadDistribute = []
# 开始
print("start ")
for minute in range(trainLength, dataLength):
    print(f"minute: {minute}")
    print(f"workload: {totalWorkloads[minute]}")
    for serviceName, service in servicesDict.items():
        if service.compute_basic_latency() < service.slo[1]:
            map[service.workloadsDict["Total"]] = service.pod
    for serviceName, service in servicesDict.items():
        if service.compute_basic_latency() < service.slo[1]:
            map[service.workloadsDict["Total"]] = service.pod
        # 获取输入数据
        tmp_data = workloadScalerDict[serviceName].transform(totalWorkloads[minute - inputLength: minute].reshape(-1, 1)).reshape(-1)
        input_data = [split_data(tmp_data)]
        # 工作负载预测
        predictedWorkload = int(workloadScalerDict[serviceName].inverse_transform(workloadPredictors[serviceName].predict(np.array(input_data)))[0][0])
        if len(service.workloadPredictErrorHistory) > 0:
            predictedWorkload = int(predictedWorkload + np.mean(service.workloadPredictErrorHistory))
        #     np.mean(service.workloadPredictErrorHistory
        # 预测工作负载
        service.workloadsDict["Total"] = predictedWorkload
        # 记录工作负载情况
        predictWorkloadDistribute.append(predictedWorkload)
        workloadDistribute.append(totalWorkloads[minute])
        slope = (totalWorkloads[minute] - totalWorkloads[minute - k]) / k
        service.slopeList.append(slope)
        if predictedWorkload in map:
            service.podEstimate = map[predictedWorkload]
        # 资源估计
        else:
            service.podEstimate = math.ceil(resourceEstimatorsDict[serviceName].predict([[service.workloadsDict["Total"], service.slo[0]]])[0])
        # 突发感知
        burst = burst_detect(service, totalWorkloads, minute, totalWorkloads[minute - 1: minute] -
                                                         totalWorkloads[minute - 2: minute - 1], kw, slope)
        if burst:
            print("burst")
            podList = service.podScaleHistory
            podList.append(service.podEstimate)
            service.podEstimate = get_max_pod(podList)
            service.burst = True
        elif not burst and service.burst:
            podList = service.podScaleHistory
            podList.append(service.podEstimate)
            service.podEstimate = get_max_pod(podList)
            service.burst = False
        print(f"serviceName: {serviceName}")
        print(f"predict workloads: {service.workloadsDict['Total']}")
        print(f"actual workloads: {totalWorkloads[minute]}")
        if service.pod != service.podEstimate:
            service.operations += 1
        service.pod = service.podEstimate
        predictedWorkload = service.workloadsDict["Total"]
        service.workloadsDict["Total"] = totalWorkloads[minute]
        print(f"latency: {service.compute_basic_latency()}")
        latencyDistribute.append(servicesDict[serviceName].compute_basic_latency())
        if service.compute_basic_latency() > service.slo[0]:
            service.violate += 1
        service.workloadsDict["Total"] = predictedWorkload
        service.add_workload_history(totalWorkloads[minute])
        service.add_workload_predict_error_history(totalWorkloads[minute])
        servicesDict[serviceName].add_workload_imbalance(totalWorkloads[minute: minute + 1] -
                                                         totalWorkloads[minute - 1: minute])

    print("---------------------------------------")
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
    print("---------------------------------------")
    tmp_data2 = []

    apiLatencyData.append(tmp_data2)
print("my method: ")
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

    # file_path = "mymethod/mymethod_world_cup_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    #
    # file_path = "mymethod/mymethod_world_cup_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')

    # file_path = "mymethod/mymethod_wiki_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    # file_path = "mymethod/mymethod_wiki_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')
    # print("内容已保存到文件:", file_path)

    # file_path = "mymethod/mymethod_s1_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    # file_path = "mymethod/mymethod_s1_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')
    # print("内容已保存到文件:", file_path)

    # 指定要保存到的文件路径
    # file_path = "mymethod/mymethod_s2_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    #
    # file_path = "mymethod/mymethod_s2_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')
    #
    # print("内容已保存到文件:", file_path)

    # file_path = "mymethod/workload/1998WorldCup1_actualWorkload.txt"
    # np.savetxt(file_path, workloadDistribute, fmt='%d')
    # file_path = "mymethod/workload/1998WorldCup1_predictWorkload.txt"
    # np.savetxt(file_path, predictWorkloadDistribute, fmt='%d')

    # print("内容已保存到文件:", file_path)
    # file_path = "mymethod/workload/s2_actualWorkload.txt"
    # np.savetxt(file_path, workloadDistribute, fmt='%d')
    # file_path = "mymethod/workload/s2_predictWorkload.txt"
    # np.savetxt(file_path, predictWorkloadDistribute, fmt='%d')
    # print("内容已保存到文件:", file_path)

    # file_path = "hpa/hpa_s3_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    #
    # file_path = "hpa/hpa_s3_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')
    #
    # print("内容已保存到文件:", file_path)



#
# # 算法名称
# methodName = "myMethod"
# # 数据集名称
# dataSetName = "worldCup"
# # save_experiment_result_of_slo_violate_record(sloViolateDataList,
# #                                              f"sloViolateRecord/{methodName}/{dataSetName}_SLO_VIOLATE.xlsx")
# # save_experiment_result_of_cost_record(costDataList, f"costRecord/{methodName}/{dataSetName}_COST.xlsx")
# # save_experiment_result_of_api_latency(apiLatencyData, f"latencyRecord/{methodName}/{dataSetName}_LATENCY.xlsx")
# # save_experiment_result_of_pod_scale(microservicesPodData, f"podRecord/{methodName}/{dataSetName}_POD.xlsx")
# # save_experiment_result_of_scale_operations(microservicesPodData, f"podRecord/{methodName}/{dataSetName}_POD.xlsx")

