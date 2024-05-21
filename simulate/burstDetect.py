import matplotlib.pyplot as plt
import pandas as pd


time_sequence_length = 10

def draw(data, text):
    """
    按照时间绘制数据，突发点为红色
    """
    data1=data.copy()

    # 使用matplotlib绘图
    plt.figure(figsize=(20, 5),dpi=300)

    # timestamp：1998-06-09 22:00:00转为分钟
    data1['minute'] = range(len(data1))

    # 画所有的requests与时间的关系
    plt.plot(data1['minute'], data1['predict'], label='Requests', color='blue', linewidth=2)

    # 找出突发的数据点
    bursts = data1[data1['burst'] == 1]

    # 画突发点
    plt.scatter(bursts['minute'], bursts['predict'], color='red', label='Bursts',s=20)

    # 设定y坐标为0以上
    plt.ylim(0, max(data1['predict']) + 1000)

    # 设置注释
    plt.title(text)

    plt.legend()
    plt.show()


def my_method(data, window_size):
    # 计算每秒请求的变化值，即每秒请求量减去前一秒请求量的绝对值
    data['variate'] = data['predict'].diff().abs()
    # 计算滑动窗内变化值的平均值
    avg = data['variate'].rolling(window=window_size).mean()
    # avg=data['variate'].mean()

    # 计算变化值的标准差
    std = data['variate'].rolling(window=window_size).std()
    # std = data['variate'].std()


    # 对于数据前长度不足滑动窗口的数据，取数据前所有数据的平均值、标准差
    for i in range(window_size):
        avg.iloc[i] = data['variate'].iloc[:i].mean()
        std.iloc[i] = data['variate'].iloc[:i].std()

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
    draw(data, 'my_method')
    data.drop(['variate'], axis=1, inplace=True)
    return data


data = pd.read_csv('fc/estimate1.csv')
# 复制数据
# for i in range(1, 15):
#     window_average(data.copy(), i*10)
data= my_method(data.copy(), 60)
# fft(data.copy())

# 保存data
data.to_csv('fc/burst1.csv', index=False)
