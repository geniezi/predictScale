import numpy as np
import pandas as pd
# from simple_pid import PID
from base_model import Cost
from model.data.workload.dataInitial import get_workload, dataLength, trainLength, \
    servicesDict, costDict, sloViolateDict, apiLatencyData, microservicesPodData, latency_latency


class PID:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def update(self, measured_value):
        error =  measured_value

        # 计算积分项
        self.integral += 0

        # 计算微分项
        derivative = 0

        # 计算输出
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        return output

class ShowAR:
    def __init__(self, serviceDict):
        self.services = []
        self.controller_map = {}
        self.serviceDict = serviceDict
        for msName, ms in serviceDict.items():
            self.services.append(ms)
            # [0.81969191 3.88647558 3.23533929]
            # [8.40596656 9.10158042 2.8509933 ]
            # [0.67372054 6.39118226 9.90680277]
            # [0.91481533 5.17549739 6.77222808]
            self.controller_map[msName] = PID(Kp=1, Ki=0, Kd=0, setpoint=ms.slo[0])

        self.beta = 0.1
        self.alpha = 0.05
        # # self.alpha = 0.32 # world cup
        self.alpha2 = 0.2
        self.times = 0
    def PID_score(self, ms):
        latency = ms.compute_basic_latency()
        if np.isnan(latency):
            return ms.slo[0]
        # calculate PID score
        pid = self.controller_map[ms.serviceName]
        output = pid.update(latency)
        print(f"latency: {latency}")
        print(f"service  output: {output}")
        # if latency > ms.slo[0] and output > 0:
        #     return output
        return output
    def horizontal_scale(self):
        ms_score_map = {}
        for ms in self.services:
            ms_score_map[ms.serviceName] = self.PID_score(ms)
        ranks = sorted(ms_score_map.items(), key=lambda x: x[1], reverse=True)
        for pair in ranks:
            ms = self.serviceDict[pair[0]]
            output = pair[1]
            RM = ms.pod
            # print(f"service {pair[0]} output: {output}")
            # print(f"service {pair[0]} latency: {ms.compute_basic_latency()}")
            if output > ms.slo[0] * (1 + self.alpha / 2):
                # self.alpha / 2
                RM = int(RM + max(1, RM * self.beta))
            elif output < ms.slo[0] * (1 - self.alpha2 / 2):
                if self.times == 0:
                    RM = int(RM - max(1, RM * self.beta))
                    self.times = 5
                else:
                    self.times -= 1
            else:
                continue
            if ms.podThreshold[0] <= RM <= ms.podThreshold[1]:
                ms.pod = RM
                # if RM != ms.pod:
                #     ms.operations += 1
            # if RM <= ms.podThreshold[0]:
            #     RM = ms.podThreshold[0]
            # elif RM > ms.podThreshold[1]:
            #     RM = ms.podThreshold[1]
            # if RM != ms.pod:
            #     ms.pod = RM
            #     ms.operations += 1

showAr = ShowAR(servicesDict)
basePath, filePath, workloads = get_workload()
latencyDistribute = []
podDistribute = []
for minute in range(0, dataLength - trainLength):
    print(f"minute: {minute}")
    print(f"workload: {workloads[minute]}")
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
    for serviceName in servicesDict.keys():
        servicesDict[serviceName].workloadsDict["Total"] = workloads[minute]
        print(f"service{serviceName} before Pod: {servicesDict[serviceName].pod}")
        pod = servicesDict[serviceName].pod
    showAr.horizontal_scale()
    for serviceName in servicesDict.keys():
        if pod != servicesDict[serviceName].pod:
            servicesDict[serviceName].operations += 1
            print("scaled")
    for serviceName in servicesDict.keys():
        print(f"service{serviceName} before Pod: {servicesDict[serviceName].pod}")
    print("-------------------")

    # 遍历每个微服务
    for serviceName in servicesDict.keys():
        latencyDistribute.append(servicesDict[serviceName].compute_basic_latency())
        if servicesDict[serviceName].compute_basic_latency() > servicesDict[serviceName].slo[0]:
            servicesDict[serviceName].violate += 1
            # 扩容延迟
            latencyDistribute.append(servicesDict[serviceName].compute_basic_latency())
            microservicesPodData.append(tmpData)

print("SHOWAR: ")

sloViolateDataList = []
costDataList = []
operationsCostDataList = []
for requestType in sloViolateDict.keys():
    print(f"{requestType} violate SLO times: {sloViolateDict[requestType]}")
    sloViolateDataList.append(sloViolateDict[requestType])
print("\n")
for serviceName in servicesDict.keys():
    print(f"{serviceName} violate SLO times: {servicesDict[serviceName].violate}")
print("\n")
for serviceName in  servicesDict.keys():
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

    # file_path = "showar/showar_world_cup_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    #
    # file_path = "showar/showar_world_cup_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')

    file_path = "showar/showar_wiki_latency.txt"
    # 将整数数组保存到文本文件
    np.savetxt(file_path, latencyDistribute, fmt='%d')

    file_path = "showar/showar_wiki_pod.txt"
    # 将整数数组保存到文本文件
    np.savetxt(file_path, podDistribute, fmt='%d')

    # file_path = "showar/showar_s1_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    #
    # file_path = "showar/showar_s1_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')

    # file_path = "showar/showar_s2_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    #
    # file_path = "showar/showar_s2_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')


# 算法名称
# methodName = "showar"
# # 数据集名称
# dataSetName = "worldCup"
# save_experiment_result_of_slo_violate_record(sloViolateDataList, f"sloViolateRecord/{methodName}/{dataSetName}_SLO_VIOLATE.xlsx")
# save_experiment_result_of_cost_record(costDataList, f"costRecord/{methodName}/{dataSetName}_COST.xlsx")
# save_experiment_result_of_api_latency(apiLatencyData, f"latencyRecord/{methodName}/{dataSetName}_LATENCY.xlsx")
# save_experiment_result_of_pod_scale(microservicesPodData, f"podRecord/{methodName}/{dataSetName}_POD.xlsx")
# save_experiment_result_of_scale_operations(microservicesPodData, f"podRecord/{methodName}/{dataSetName}_POD.xlsx")
