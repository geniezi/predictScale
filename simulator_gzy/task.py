from node import Node


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
        self.status = 'waiting'  # waiting, running, completed

        # 新增属性
        self.start_run_time = None
        self.end_run_time = None
        self.prefix_time = 0
        self.speedup = 1
        self.is_assigned = False

    def update_status(self, node, current_time):
        if self.status == 'waiting' and self.is_assigned:
            self.start_run_time = current_time
            self.status = 'running'
            print(f"Task {self.task_name} (id: {self.task_id}) has started on node {node.node_id}.")

        if self.status == 'running' and (
                (current_time - self.start_run_time) * self.speedup + self.prefix_time >= self.run_time):
            self.end_run_time = current_time
            self.status = 'completed'
            print(f"Task {self.task_name} (id: {self.task_id}) has completed on node {node.node_id}.")

    def get_waiting_time(self):
        # Task's waiting time is the difference between the actual start time and the planned start time
        if self.start_run_time:
            return self.start_run_time - self.start_time
        else:
            return None

    # Call this method to get a summary of a task's performance data
    def get_performance_data(self):
        waiting_time = self.get_waiting_time()
        run_time = self.run_time if self.status == 'completed' else None
        return {
            'task_id': self.task_id,
            'start_time': self.start_time,
            'start_run_time': self.start_run_time if self.start_run_time else 'Not started yet',
            'end_run_time': self.end_run_time if self.end_run_time else 'Not completed yet',
            'waiting_time': waiting_time,
            'run_time': run_time,
        }

    def assign(self, node, current_time):
        self.is_assigned = True
        if self.gpu_type != '':
            if self.gpu_type == 'T4':
                self.speedup = 1.5 if node.gpu_type == 'P100' else 2 if node.gpu_type == 'V100' else 1
            elif self.gpu_type == 'P100':
                self.speedup = 4 / 3 if node.gpu_type == 'V100' else 2 / 3 if node.gpu_type == 'T4' else 1
            elif self.gpu_type == 'V100':
                self.speedup = 3 / 4 if node.gpu_type == 'P100' else 0.5 if node.gpu_type == 'T4' else 1
        self.update_status(node, current_time)

    def get_status(self):
        return self.status
