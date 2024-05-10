import matplotlib.pyplot as plt
import numpy as np

import util.osUtil as osUtil


def average(data, window_size):
    """
    计算数据的滑动平均值
    :param data: 数据
    :param window_size: 窗口大小
    :return: 滑动平均值
    """
    avg = []
    for i in range(len(data)):
        if i < window_size:
            avg.append(sum(data[:i + 1]) / (i + 1))
        else:
            avg.append(sum(data[i - window_size + 1:i + 1]) / window_size)
    return avg


def burst(data, avg):
    """
    判断数据是否为突发
    :param data: 数据
    :param avg: 平均值
    """
    mul=[]
    for i in range(len(data)):
        mul.append(0.5)
    avg_series = np.array(avg)
    data['burst'] = (abs(data['requests'] - avg_series) > 0.05*avg_series).astype(int)


def draw(data):
    """
    按照时间绘制数据，突发点为红色
    :param data: 数据
    """
    # 使用matplotlib绘图
    plt.figure(figsize=(20, 5))

    # 画所有的requests与时间的关系
    plt.plot(data['minute'], data['requests'], label='Requests', color='blue')

    # 找出突发的数据点
    bursts = data[data['burst'] == 1]

    # 画突发点
    plt.scatter(bursts['minute'], bursts['requests'], color='red', label='Bursts')

    plt.legend()
    plt.show()

def main():
    data = osUtil.read_data('pageviews_by_minute.tsv')
    data1 = data['requests']
    window_size = 5
    avg = average(data1, window_size)
    # 根据前一刻的平均值，判断当前时刻是否为突发
    burst(data, avg)

    draw(data)

if __name__ == '__main__':
    main()