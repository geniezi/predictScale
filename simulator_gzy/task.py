import config


class Task:
    def __init__(self, task_id, task_name, start_time, run_time, plan_cpu, plan_mem, plan_gpu, gpu_type, task_type):
        self.task_id = task_id
        self.task_name = task_name
        self.start_time = start_time
        self.run_time = run_time
        self.plan_cpu = plan_cpu
        self.plan_mem = plan_mem
        self.plan_gpu = plan_gpu
        self.gpu_type = gpu_type
        self.task_type = task_type
        self.node = None
        self.status = 'waiting'  # waiting, running, completed

        # 新增属性
        self.start_run_time = None
        self.end_run_time = None
        self.prefix_time = 0
        self.speedup = 1
        self.is_assigned = False

    def update_status(self, current_time):
        if self.status == 'waiting' and self.is_assigned:
            self.start_run_time = current_time
            self.status = 'running'
            print(f"Task {self.task_name} (id: {self.task_id}) has started on node {self.node.node_id}.")

        if self.status == 'running' and (
                (current_time - self.start_run_time) * self.speedup + self.prefix_time >= self.run_time):
            self.end_run_time = current_time
            self.status = 'completed'
            print(f"Task {self.task_name} (id: {self.task_id}) has completed on node {self.node.node_id}.")
            self.node.free_resources(self)

    def get_waiting_time(self, current_time):
        # Task's waiting time is the difference between the actual start time and the planned start time
        if self.start_run_time:
            return self.start_run_time - self.start_time
        else:
            return current_time - self.start_time

    # Call this method to get a summary of a task's performance data
    def get_performance_data(self, current_time):
        return {
            config.SAVE_FIELDS[0]: self.task_id,
            config.SAVE_FIELDS[8]: self.node.node_id if self.node else None,
            config.SAVE_FIELDS[1]: self.start_time,
            config.SAVE_FIELDS[2]: self.run_time,
            config.SAVE_FIELDS[9]: self.status,
            config.SAVE_FIELDS[3]: self.start_run_time,
            config.SAVE_FIELDS[4]: self.end_run_time,
            config.SAVE_FIELDS[5]: self.get_waiting_time(current_time),
            config.SAVE_FIELDS[6]: self.prefix_time,
            config.SAVE_FIELDS[7]: self.speedup,
        }

    def assign(self, node, current_time):
        self.is_assigned = True
        self.node = node
        if self.gpu_type != '':
            if self.gpu_type == 'MISC':
                self.speedup = config.gpuConfig[node.gpu_type]
            elif self.gpu_type == 'T4':
                self.speedup = config.gpuConfig[node.gpu_type] / 2
            elif str.__contains__(self.gpu_type, '100'):
                self.speedup = config.gpuConfig[node.gpu_type] / 3
        self.update_status(current_time)

    def get_status(self):
        return self.status
