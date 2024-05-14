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

    # 设置注释
    plt.title(text)

    plt.legend()
    plt.show()


def variate(data, window_size):
    data1=data.copy()
    # 计算每秒请求的变化值，即每秒请求量减去前一秒请求量的绝对值
    data1['variate'] = data1['predict'].diff().abs()
    for i in range(time_sequence_length):
        if i==0:
            data1.loc[data1.index[i], 'variate'] = 0
        else:
            data1.loc[data1.index[i], 'variate']=abs(data1['predict'].iloc[i]-data1['requests'].iloc[i-1])

    # 计算滑动窗内变化值的平均值
    # avg = data1['variate'].rolling(window=window_size).mean()
    avg=data1['variate'].mean()

    # 计算变化值的标准差
    # std = data1['variate'].rolling(window=window_size).std()
    std = data1['variate'].std()
    # 计算变化值的阈值
    threshold = [avg - 3 * std, avg + 3 * std]
    # threshold = []
    # for i in range(len(data1)):
    #     if i < window_size:
    #         threshold.append([0,0])
    #     else:
    #         threshold.append([avg.iloc[i] - 3 * std.iloc[i], avg.iloc[i] + 3 * std.iloc[i]])

    # 判断是否为突发点
    burst = []
    for i in range(len(data1)):
        if i < window_size:
            burst.append(0)
        else:
            if data1['variate'].iloc[i] > threshold[1] or data1['variate'].iloc[i] < threshold[0]:
            # if data1['variate'].iloc[i] > threshold[i][1] or data1['variate'].iloc[i] < threshold[i][0]:
                burst.append(1)
            else:
                burst.append(0)
    data1['burst'] = burst
    draw(data1, 'variate')
    data1.drop(['variate'], axis=1, inplace=True)
    return data1


data = pd.read_csv('estimate.csv')
# 复制数据
# for i in range(1, 15):
#     window_average(data.copy(), i*10)
data= variate(data.copy(), 60)
# fft(data.copy())

# 保存data
data.to_csv('burst.csv', index=False)
