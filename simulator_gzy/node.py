class Node:
    def __init__(self, node_id, num_cpu=96, total_mem=720, num_gpu=8, gpu_type='CPU'):
        self.node_id = node_id
        self.total_cpu = num_cpu * 100  # 以百分比表示
        self.total_mem = total_mem  # 以GB为单位
        self.total_gpu = num_gpu * 100  # 以百分比表示
        self.gpu_type = gpu_type
        self.available_cpu = self.total_cpu
        self.available_mem = self.total_mem
        self.available_gpu = self.total_gpu
        self.running_tasks = []  # 运行中的任务

    def check_if_assignable(self, task):
        # 检查是否有足够的资源来运行任务
        return (self.available_cpu >= task.plan_cpu and
                self.available_mem >= task.plan_mem and
                self.available_gpu >= task.plan_gpu)

    def assign_task(self, task, current_time):
        # 任务分配的逻辑
        if self.check_if_assignable(task):

            self.available_cpu -= task.plan_cpu
            self.available_mem -= task.plan_mem
            self.available_gpu -= task.plan_gpu
            self.running_tasks.append(task)
            task.assign(self, current_time)
        else:
            print("Not enough resources to run the task on this node.")

    def free_resources(self, task):
        # 释放资源
        self.available_cpu += task.plan_cpu
        self.available_mem += task.plan_mem
        self.available_gpu += task.plan_gpu
        self.running_tasks.remove(task)

    def get_completed_tasks(self, current_time):
        # 遍历该节点上正在运行的任务
        task_list = []
        for task in self.running_tasks:
            # 假设Task类有一个方法来判断任务是否完成
            task.update_status(self,current_time)
            if task.get_status() == "completed":
                task_list.append(task)
        return task_list
