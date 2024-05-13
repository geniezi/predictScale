import numpy as np
import pandas as pd
import tensorflow as tf
from keras.engine.base_layer import Layer
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 神经单元
units = 4
units2 = 32
# 修改注意力层的参数
attention_units = 8  # 增加注意力权重的维度
# 输出步长
future_steps = 1
# 输入步长
time_sequence_length = 10

filePath = "世界杯数据集Day46.xlsx"


# filePath = "维基百科数据集桌面端原始版.xlsx"
# filePath = "维基百科数据集桌面端缩放版.xlsx"
# filePath = "维基百科.xlsx"




def load_weight():
    class AdditiveAttention(Layer):
        def __init__(self, attention_units=1, **kwargs):
            self.attention_units = attention_units
            super(AdditiveAttention, self).__init__(**kwargs)

        def build(self, input_shape):
            # Get the last dimension of the input shape
            input_dim = input_shape[-1]

            # Initialize weights
            self.v = self.add_weight(name="v",
                                     shape=(self.attention_units, 1),  # 修改此处的形状
                                     initializer="random_normal",
                                     trainable=True)
            self.w_query = self.add_weight(name="w_query",
                                           shape=(input_dim, self.attention_units),
                                           initializer="random_normal",
                                           trainable=True)
            self.w_key = self.add_weight(name="w_key",
                                         shape=(input_dim, self.attention_units),
                                         initializer="random_normal",
                                         trainable=True)
            super(AdditiveAttention, self).build(input_shape)

        def call(self, inputs):
            # Calculate attention scores
            query = tf.matmul(inputs, self.w_query)
            key = tf.matmul(inputs, self.w_key)
            tanh_output = tf.nn.tanh(query + key)
            score = tf.matmul(tanh_output, self.v)  # 不需要进行转置操作
            attention_weights = tf.nn.softmax(score, axis=1)

            # Weighted sum of inputs
            weighted_inputs = inputs * attention_weights
            output = tf.reduce_sum(weighted_inputs, axis=1)

            return output

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[-1])

        def get_config(self):
            config = super(AdditiveAttention, self).get_config()
            config.update({
                'attention_units': self.attention_units
            })
            return config

    # 重建和之前相同的模型架构
    new_model = Sequential()
    new_model.add(LSTM(units=units, input_shape=(time_sequence_length, 1), return_sequences=True))
    new_model.add(LSTM(units=units2, return_sequences=True))
    new_model.add(AdditiveAttention(attention_units=attention_units))
    new_model.add(Dense(units=future_steps))
    # 在你编译模型之后，加载权重
    learning_rate = 0.05
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
    new_model.compile(optimizer=optimizer, loss='mean_squared_error')
    # 加载之前保存的权重
    new_model.load_weights('model_weights.h5')
    return new_model


def gru_prepare_data(data, time_steps):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
    return np.array(X)


model = load_weight()
df_all = pd.read_excel(filePath, header=None, engine='openpyxl',names=['timestamp','requests'])
df = df_all['requests']
data = df.values
scaler = MinMaxScaler()
# 原始值
dataNoScaled = np.array(data).reshape(-1, 1)
# 归一化数据
dataScaled = scaler.fit_transform(dataNoScaled)

X_val = gru_prepare_data(dataScaled, time_sequence_length)
# X_predict=np.reshape(X_predict,(X_predict.shape[0],X_predict.shape[1],1))
predictScaled = model.predict(X_val)
predict = scaler.inverse_transform(predictScaled)
predict = predict.reshape(-1)
# 转换成pd
df1 = pd.DataFrame(predict)
# 四舍五入取整
df1 = df1.round()
df1.columns = ['predict']
# 添加到原始数据中，前time_sequence_length个值为0
df_all['predict'] = 0  # 创建一个新列填充了NaN
df_all.loc[time_sequence_length:, 'predict'] = df1['predict'].values
df_all['predict'] = df_all['predict'].astype(int)
# 保存到csv
df_all.to_csv('../predict.csv', index=False)
# print(df)