import numpy as np
from base_model import Cost
from bayes_opt import BayesianOptimization, UtilityFunction
from model.data.workload.dataInitial import get_workload, dataLength, trainLength, \
    servicesDict, costDict, sloViolateDict, microservicesPodData


class MicroScaler:
    def __init__(self, servicesDict):
        self.p_min = 0.7
        self.p_max = 0.83

        self.n_iter = 3
        self.mss = []
        self.services = servicesDict
        for serviceName in servicesDict.keys():
            self.mss.append(servicesDict[serviceName])
        self.up = set()
        self.down = set()
        # self.times = 10
        # self.times = 12 wiki

    # Cost of container
    def price(self, pod_count):
        return pod_count

    # p=P50/P90
    def p_value(self, svc):
        print(f"serviceName: {svc.serviceName}")
        p90 = svc.compute_latency_of_percentile(0.9)
        print(f"p90: {p90}")
        p50 = svc.compute_latency_of_percentile(0.5)
        print(f"p50: {p50}")
        if p90 == 0:
            return np.NaN
        else:
            return float(p50) / float(p90)

    def detector(self):
        svcs = self.mss
        svc_count_dic = {}
        for svc in svcs:
            svc_count_dic[svc.serviceName] = svc.pod
        [print(svc, svc_count_dic[svc]) for svc in svc_count_dic.keys() if svc_count_dic[svc] != 1]
        # Detect abnormal services and obtain the abnormal service list
        ab_svcs = []
        for svc in svcs:
            t = svc.compute_latency_of_percentile(0.9)
            if t >= svc.slo[0]:
                ab_svcs.append(svc)
            if t < svc.slo[0]:
                ab_svcs.append(svc)
        self.service_power(ab_svcs)

    # Decide which services need to scale
    def service_power(self, ab_svcs):
        for ab_svc in ab_svcs:
            print(f"ab_svc: {ab_svc.serviceName}")
        for ab_svc in ab_svcs:
            p = self.p_value(ab_svc)
            # p = ab_svc.compute_basic_latency()
            if np.isnan(p):
                continue
            # p > self.p_max and
            if p > self.p_max:
                self.down.add(ab_svc)
            # elif p > self.p_max:
            else:
                self.up.add(ab_svc)

    # Auto-scale Decision
    def auto_scale(self):
        # print("so: ")
        # for svc in self.up:
        #     print(svc.serviceName)
        # print("si: ")
        # for svc in self.down:
        #     print(svc.serviceName)
        for svc in self.up:
            # scale up
            origin_pod_count = svc.pod
            if origin_pod_count == 6:
                # svc.podThreshold[1]
                continue
            index = self.mss.index(svc)
            pbounds = {'x': (origin_pod_count, 6), 'index': [index, index]}
            # svc.pod += 1
            self.BO(self.scale, pbounds)
            if origin_pod_count != svc.pod:
                svc.operations += 1
        for svc in self.down:
            # scale down
            origin_pod_count = svc.pod
            index = self.mss.index(svc)
            if origin_pod_count == 1:
                continue
            pbounds = {'x': (1, origin_pod_count), 'index': [index, index]}
            # svc.pod -= 1
            self.BO(self.scale, pbounds)
            if origin_pod_count != svc.pod:
                svc.operations += 1
            # t = threading.Thread(target=self.BO, args=(self.scale, pbounds))
            # t.setDaemon(True)
            # t.start()
        self.up.clear()
        self.down.clear()

    def scale(self, x, index):
        svc = self.mss[int(index)]
        svc.pod = int(x)
        print('{} is scaled to {}'.format(svc.serviceName, int(x)))
        svcs_counts = {}
        for svc in self.mss:
            svcs_counts[svc.serviceName] = svc.pod
        for svc in svcs_counts.keys():
            P90 = self.services[svc].compute_latency_of_percentile(0.9)
            P90 = self.services[svc].compute_basic_latency()
            score = 0
            if P90 > self.services[svc].slo[0]:
                score = -P90 * self.price(svcs_counts[svc]) - P90 * 10
            else:
                score = -P90 * self.price(svcs_counts[svc]) + P90 * 10
            return score

    # Bayesian optimization

    # def BO(self, f, pbounds):
    #     optimizer = BayesianOptimization(
    #         f=f,
    #         pbounds=pbounds,
    #         random_state=1,
    #         allow_duplicate_points=True  # 允许重复的数据点
    #     )
    #
    #     # 设置高斯过程参数
    #     optimizer.set_gp_params(kernel=None)
    #
    #     # 创建 UtilityFunction 的实例
    #     utility = UtilityFunction(kind="ucb", kappa=2.576, xi=0.0)
    #
    #     # 使用 acquisition_function 参数传递 UtilityFunction 实例
    #     optimizer.maximize(
    #         init_points=2,
    #         n_iter=self.n_iter,
    #         acquisition_function=utility,  # 修改这里
    #     )
    #
    #     return optimizer.max["params"]["x"]
    def BO(self, f, pbounds):
        optimizer = BayesianOptimization(
            f=f,
            pbounds=pbounds,
            random_state=1,
            allow_duplicate_points=True  # 允许重复的数据点
        )

        # 设置高斯过程参数
        optimizer.set_gp_params(kernel=None, alpha=1e-6)

        # 创建 UtilityFunction 的实例
        utility = UtilityFunction(kind="ucb", kappa=5, xi=0.0)  # 增加 kappa 的值 8 8 13

        # 使用 acquisition_function 参数传递 UtilityFunction 实例
        optimizer.maximize(
            init_points=5,  # 增加 init_points 的值 8 8 13
            n_iter=self.n_iter,
            acquisition_function=utility,
        )

        return optimizer.max["params"]["x"]

basePath, filePath, workloads = get_workload()
microScaler = MicroScaler(servicesDict)
latencyDistribute = []
podDistribute = []
for minute in range(0, dataLength - trainLength):
    tmpData = []
    for serviceName in servicesDict.keys():
        servicesDict[serviceName].workloadsDict["Total"] = workloads[minute]
        print(f"service{serviceName} Pod: {servicesDict[serviceName].pod}")
        servicesDict[serviceName].add_latency_history()
        tmpData.append(servicesDict[serviceName].pod)

        print(f"minute: {minute}")
    for serviceName in servicesDict.keys():
        print(f"service: {serviceName}")
        print(f"service latency Up: {servicesDict[serviceName].slo[0]}")
        print(f"service latency Down: {servicesDict[serviceName].slo[1]}")
        print(f"latency now: {servicesDict[serviceName].compute_basic_latency()}")
        print(f"service pod: {servicesDict[serviceName].pod}")
        print("-----------------------------------------------------")
    # 遍历每个微服务
    tmpData = []
    for serviceName in servicesDict.keys():
        latencyDistribute.append(servicesDict[serviceName].compute_basic_latency())
        if servicesDict[serviceName].compute_basic_latency() > servicesDict[serviceName].slo[0]:
            servicesDict[serviceName].violate += 1
            servicesDict[serviceName].add_latency_history()
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
    serviceNow = {}
    for serviceName in servicesDict.keys():
        serviceNow[serviceName] = servicesDict[serviceName].pod
        podDistribute.append(servicesDict[serviceName].pod)
    microScaler.detector()
    microScaler.auto_scale()
    # for serviceName in servicesDict.keys():
    #     if servicesDict[serviceName].pod != serviceNow[serviceName]:
    #         servicesDict[serviceName].operations += 1
    for serviceName in servicesDict.keys():
        print(f"service{serviceName} Pod: {servicesDict[serviceName].pod}")
    print("-------------------")

    microservicesPodData.append(tmpData)

print("MicroScaler: ")

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
    # # 指定要保存到的文件路径
    # file_path = "microscaler/microscaler_world_cup_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    #
    # file_path = "microscaler/microscaler_world_cup_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')
    #
    # print("内容已保存到文件:", file_path)

    file_path = "microscaler/microscaler_wiki_latency.txt"
    # 将整数数组保存到文本文件
    np.savetxt(file_path, latencyDistribute, fmt='%d')

    file_path = "microscaler/microscaler_wiki_pod.txt"
    # 将整数数组保存到文本文件
    np.savetxt(file_path, podDistribute, fmt='%d')

    print("内容已保存到文件:", file_path)

    # file_path = "microscaler/microscaler_s1_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    #
    # file_path = "microscaler/microscaler_s1_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')
    #
    # print("内容已保存到文件:", file_path)
    #
    # file_path = "microscaler/microscaler_s2_latency.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, latencyDistribute, fmt='%d')
    #
    # file_path = "microscaler/microscaler_s2_pod.txt"
    # # 将整数数组保存到文本文件
    # np.savetxt(file_path, podDistribute, fmt='%d')

    # print("内容已保存到文件:", file_path)
print("-------------------------")

