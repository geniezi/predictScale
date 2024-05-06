class Cluster:
    def __init__(self):
        self.nodes = []
        self.tasks_queue = []
        self.assigned_task = []
        self.waiting_tasks = []
        self.running_tasks = []
        self.task_index = 0

    def add_node(self, node):
        self.nodes.append(node)

    def add_task(self, task):
        self.tasks_queue.append(task)

    def update_started_tasks(self, current_time):
        while self.task_index < len(self.tasks_queue):
            task = self.tasks_queue[self.task_index]
            if task.start_time > current_time:
                break
            self.waiting_tasks.append(task)
            print(f"Task {task.task_name} (id: {task.task_id}) has arrived.")
            self.task_index += 1

    def time_step(self, current_time):
        # 单进程
        # 遍历全部节点，并为每个节点创建一个线程来检查任务是否已完成
        # print("check and free resources on node")
        for task in self.running_tasks:
            task.update_status(current_time)
        self.running_tasks = [task for task in self.running_tasks if task.status != 'completed']
        # print("update started tasks")
        # 更新等待任务
        if self.task_index < len(self.tasks_queue):
            self.update_started_tasks(current_time)

    def assign_task(self, task, node, current_time):
        node.assign_task(task, current_time)
        self.running_tasks.append(task)
        self.assigned_task.append(task)

    def get_first_assignable_node(self, task):
        for node in self.nodes:
            if node.check_if_assignable(task):
                return node
        return None

    def get_last_assignable_node(self, task):
        for node in reversed(self.nodes):
            if node.check_if_assignable(task):
                return node
        return None

    def shutdown(self):
        pass
