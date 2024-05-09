import subprocess
from concurrent.futures import ProcessPoolExecutor, wait

import config


# # get date list from 1970-1-1 to 1970-1-31
# end_date_list = []
# for month, day in [(1, 15),
#                    (1, 16),
#                    (1, 20),
#                    (1, 27),
#                    (2, 3),
#                    (2, 10),
#                    (2, 17),
#                    (2, 24),
#                    ]:
#     end_date_list.append("1970-" + str(month) + "-" + str(day))
# for num_nodes in [50, 40, 30]:
#     for end_date in end_date_list:
#         for scheduler in config.scheduleStrategyConfig.keys():
#             os.system(
#                 "python simulator_gzy/simulator.py --scheduler {} --start_date {} --end_date {} --num_nodes {}".format(
#                     scheduler, "1970-01-13", end_date, num_nodes))


# 函数来执行外部命令
def execute_command(scheduler, start_date, end_date, num_nodes):
    command = f"python simulator_gzy/simulator.py --scheduler {scheduler} --start_date {start_date} --end_date {end_date} --num_nodes {num_nodes}"
    # Popen不会阻塞，允许我们同时运行多个进程
    process = subprocess.Popen(command, shell=True)

    return process


def main():
    # 创建一个ProcessPoolExecutor来并行执行命令
    with ProcessPoolExecutor(2) as executor:
        futures = []
        for month, day in [(1, 15),
                           # (1, 16),
                           # (1, 20),
                           # (1, 27),
                           # (2, 3),
                           # (2, 10),
                           # (2, 17),
                           # (2, 24)
                           ]:
            end_date = f"1970-{month}-{day}"
            for num_nodes in [50, 40, 30]:
                for scheduler in config.scheduleStrategyConfig.keys():
                    # 为每个任务建立Future对象，并加入列表
                    future = executor.submit(execute_command, scheduler, "1970-01-13", end_date, num_nodes)
                    futures.append(future)

        # 等待所有的进程完成
        wait(futures)
        # for future in futures:
        #     future.result()


if __name__ == '__main__':
    main()
