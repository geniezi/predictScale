import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate

import util.osUtil as osUtil


def draw(data, text):
    """
    按照时间绘制数据，突发点为红色
    :param data: 数据
    """
    # 使用matplotlib绘图
    plt.figure(figsize=(20, 5))

    # 画所有的requests与时间的关系
    plt.plot(data['minute'], data['requests_scale'], label='Requests', color='blue')

    # 找出突发的数据点
    bursts = data[data['burst'] == 1]

    # 画突发点
    plt.scatter(bursts['minute'], bursts['requests_scale'], color='red', label='Bursts')

    # 设置注释
    plt.title(text)

    plt.legend()
    plt.show()


def window_average(data, window_size):
    """
    计算数据的滑动平均值
    :param data: 数据
    :param window_size: 窗口大小
    :return: 滑动平均值
    """
    avg = []
    # 标准差
    std = []
    # 对每个数据求前window_size个数据的平均值，标准差
    for i in range(len(data)):
        if i < window_size:
            avg.append(data['requests_scale'].iloc[i])
            std.append(0)
        else:
            avg.append(data['requests_scale'].iloc[i - window_size:i].mean())
            std.append(data['requests_scale'].iloc[i - window_size:i].std())

    # 判断是否为突发点
    burst = []
    for i in range(len(data)):
        if i < window_size:
            burst.append(0)
        else:
            if abs(data['requests_scale'].iloc[i] - avg[i]) > std[i]:
                burst.append(1)
            else:
                burst.append(0)
    data['burst'] = burst

    draw(data, 'window:{}'.format(window_size))


def variate(data):
    # 计算每秒请求的变化值，即每秒请求量减去前一秒请求量的绝对值
    data['variate'] = data['requests_scale'].diff().abs()
    # 计算滑动窗内变化值的平均值
    # avg = data['variate'].rolling(window=window_size).mean()
    avg=data['variate'].mean()

    # 计算变化值的标准差
    # std = data['variate'].rolling(window=window_size).std()
    std = data['variate'].std()
    # 计算变化值的阈值
    threshold = [avg - 3 * std, avg + 3 * std]
    # threshold = []
    # for i in range(len(data)):
    #     if i < window_size:
    #         threshold.append([0, 0])
    #     else:
    #         threshold.append([avg.iloc[i] - 3 * std.iloc[i], avg.iloc[i] + 3 * std.iloc[i]])
    # 判断是否为突发点
    burst = []
    for i in range(len(data)):
        if i == 0:
            burst.append(0)
        else:
            if data['variate'].iloc[i] > threshold[1] or data['variate'].iloc[i] < threshold[0]:
            # if data['variate'].iloc[i] > threshold[i][1] or data['variate'].iloc[i] < threshold[i][0]:
                burst.append(1)
            else:
                burst.append(0)
    data['burst'] = burst
    draw(data, 'variate')


def my_method(data, window_size):
    # 计算每秒请求的变化值，即每秒请求量减去前一秒请求量的绝对值
    data['variate'] = data['requests_scale'].diff().abs()
    # 计算滑动窗内变化值的平均值
    avg = data['variate'].rolling(window=window_size).mean()
    # avg=data['variate'].mean()

    # 计算变化值的标准差
    std = data['variate'].rolling(window=window_size).std()
    # std = data['variate'].std()
    # 计算变化值的阈值
    # threshold = [avg - 3 * std, avg + 3 * std]
    threshold = []
    for i in range(len(data)):
        if i < window_size:
            threshold.append([0, 0])
        else:
            threshold.append([avg.iloc[i] - 3 * std.iloc[i], avg.iloc[i] + 3 * std.iloc[i]])
    # 判断是否为突发点
    burst = []
    for i in range(len(data)):
        if i == 0:
            burst.append(0)
        else:
            # if data['variate'].iloc[i] > threshold[1] or data['variate'].iloc[i] < threshold[0]:
            if data['variate'].iloc[i] > threshold[i][1] or data['variate'].iloc[i] < threshold[i][0]:
                burst.append(1)
            else:
                burst.append(0)
    data['burst'] = burst
    draw(data, 'variate')

def average(data):
    # 计算每秒请求的平均值
    avg = data['requests_scale'].mean()
    # 计算每秒请求的标准差
    std = data['requests_scale'].std()
    # 计算阈值
    threshold = [avg - 3 * std, avg + 3 * std]
    # 判断是否为突发点
    burst = []
    for i in range(len(data)):
        if data['requests_scale'].iloc[i] > threshold[1] or data['requests_scale'].iloc[i] < threshold[0]:
            burst.append(1)
        else:
            burst.append(0)
    data['burst'] = burst
    draw(data, 'average')


def main():
    data = osUtil.read_data('pageviews_by_minute.tsv')
    # 复制数据
    # for i in range(1, 15):
    average(data.copy())
    window_average(data.copy(), 30)
    variate(data.copy())
    my_method(data.copy(), 13)
    # fft(data.copy())


if __name__ == '__main__':
    main()
