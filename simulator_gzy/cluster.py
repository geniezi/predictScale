from multiprocessing import Pool


def update_all_tasks(task_data):
    task, current_time = task_data
    node = task.update_status(current_time)
    return node, task


def node_free_resources(node_task):
    node, task = node_task
    node.free_resources(task)


class Cluster:
    def __init__(self):
        self.nodes = []
        self.tasks_queue = []
        self.assigned_task = []
        self.waiting_tasks = []
        self.running_tasks = []
        self.task_index = 0
        self.pool = Pool()

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
        self.check_finished_tasks(current_time)

        # 更新等待任务
        if self.task_index < len(self.tasks_queue):
            self.update_started_tasks(current_time)

    def check_finished_tasks(self, current_time):
        # 遍历全部节点，并为每个节点创建一个线程来检查任务是否已完成
        # # 单进程
        # for task in self.running_tasks:
        #     task.update_status(current_time)
        # self.running_tasks = [task for task in self.running_tasks if task.status != 'completed']
        # 多进程
        if len(self.running_tasks) > 0:
            task_data_list = [(task, current_time) for task in self.running_tasks]
            # 创建一个进程池，并使用map方法来运行update_status
            node_task_list = self.pool.map(update_all_tasks, task_data_list)
            self.pool.map(node_free_resources, node_task_list) #有问题
            self.running_tasks = [task for task in self.running_tasks if task.status != 'completed']

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
