import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
            if abs(data['requests_scale'].iloc[i] - avg[i]) >std[i]:
                burst.append(1)
            else:
                burst.append(0)
    data['burst'] = burst

    draw(data, 'window:{}'.format(window_size))


def variate(data):
    # 计算每秒请求的变化值，即每秒请求量减去前一秒请求量的绝对值
    data['variate'] = data['requests_scale'].diff().abs()
    # 计算变化值的平均值
    avg = data['variate'].mean()
    # 计算变化值的标准差
    std = data['variate'].std()
    # 计算变化值的阈值
    threshold = avg + 3 * std
    # 判断是否为突发点
    burst = []
    for i in range(len(data)):
        if i == 0:
            burst.append(0)
        else:
            if data['variate'].iloc[i] > threshold:
                burst.append(1)
            else:
                burst.append(0)
    data['burst'] = burst
    draw(data, 'variate')


def fft(data):
    # 设置突发的阈值，这个阈值需要根据实际情况进行调整
    BURST_THRESHOLD = 1.5

    # 假设data是已经加载的DataFrame
    # data = pd.read_csv('your_data.csv')
    # 假设我们只对'requests'列进行操作
    requests = data['requests'].values

    # 计算自相关性
    autocorr = correlate(requests, requests, mode='full')
    # 找到自相关性最高的点（除了0延迟）
    peaks = np.where(autocorr == np.max(autocorr[len(autocorr) // 2 + 1:-1]))[0][0]

    # 确定重复模式的长度L
    L = peaks - len(requests) + 1
    L = peaks - len(requests) // 2
    window_size = max(L, 1)

    # 假设我们用简单的移动平均值作为预测模型
    # 你可以使用你的AR模型或其他模型替换这里
    window_size = L
    moving_avg_prediction = np.convolve(requests, np.ones(window_size) / window_size, 'same')

    # 计算残差，即真实值和预测值之间的差异
    residuals = requests - moving_avg_prediction

    # 突发检测
    bursts = []
    for i in range(len(requests)):
        if abs(residuals[i]) > BURST_THRESHOLD * np.std(residuals):
            bursts.append(1)  # 突发发生
        else:
            bursts.append(0)  # 没有突发

    # 将检测到的突发情况添加到原始的DataFrame中
    data['burst'] = bursts

    # 输出含有突发检测结果的DataFrame
    draw(data, 'fft')


def main():
    data = osUtil.read_data('pageviews_by_minute.tsv')
    # 复制数据
    for i in range(1, 15):
        window_average(data.copy(), i*10)
    # variate(data.copy())
    # fft(data.copy())


if __name__ == '__main__':
    main()
