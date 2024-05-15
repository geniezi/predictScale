import csv
import importlib
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.engine.base_layer import Layer
from keras.layers import LSTM, GRU
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 总数据集长度
totalLength = int(1440 * 7 / 0.8)
# 训练集长度
trainDataLength = int(totalLength * 0.8)
# 验证集长度
validDataLength = int(totalLength * 0.1)
# 测试集结尾
testDataLength = totalLength
# 特征数量
features = 1
# 训练次数
epochs = 3500
# 批处理大小
batchSize = 1024
# 神经单元
units = 16
units2 = 16
# 修改注意力层的参数
attention_units = 2  # 增加注意力权重的维度
# 输出步长
future_steps = 1
# 输入步长
time_sequence_length = 10
# 残差长度
residualLength = 100
# 早停次数
early_stopping_times = 3500
# 定义L2正则化的参数
l2_regularization = 0.01  # 调整正则化强度的参数
filePath = "世界杯数据集Day46.xlsx"


# filePath = "维基百科数据集桌面端原始版.xlsx"
# filePath = "维基百科数据集桌面端缩放版.xlsx"
# filePath = "维基百科.xlsx"

def rmse_and_mse_mae_compute(actualData, predictData, modelName):
    print(f"{modelName}:")
    print(f"Input: {time_sequence_length}\nOutput: {future_steps}")
    print(f"epochs:{epochs}\nbatchSize:{batchSize}\nunits:{units}\nunits2:{units2}\nattention_units:{attention_units}")
    mse = mean_squared_error(actualData, predictData)
    print("MSE:", mse)
    # 计算均方根误差（RMSE）
    rmse = math.sqrt(mse)
    print("RMSE:", rmse)
    # 计算r2
    r2 = r2_score(actualData, predictData)
    print("r2:", r2)
    # 计算校正决定系数R2
    ar2 = 1 - ((1 - r2) * (len(actualData) - 1)) / (len(actualData) - features - 1)
    print("校正r2:", ar2)
    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(actualData, predictData)
    print("MAE:", mae)
    return [rmse, mse, mae, ar2]


# 线性回归
def new_linearregression(dataScaled):
    def linearregression_prepare_data(data, seq_length):
        X, y = [], []
        # 循环将数据保存在x中
        for i in range(0, len(data) - future_steps):
            X.append(data[i][0])
        # 如果数据长度大于时间窗口，那么把大于部分用作标签
        if len(data) > time_sequence_length:
            for times in range(1, future_steps + 1):
                y.append(data[seq_length: seq_length + times][0][0])
        # 否则数据仍然认为是输入部分
        else:
            X.append(data[seq_length - future_steps: seq_length][0][0])
        return np.array([X]), np.array([y])

    def new_linearregression_train(X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    # np.random.seed(42)
    # tf.random.set_seed(42)
    predictData = []
    print("测试长度:", testDataLength - trainDataLength - validDataLength)
    for step in range(trainDataLength + validDataLength, testDataLength - future_steps, future_steps):
        X_train, y_train = linearregression_prepare_data(dataScaled[step - time_sequence_length - future_steps: step],
                                                         time_sequence_length)
        model = new_linearregression_train(X_train, y_train)
        X_train, y_train = linearregression_prepare_data(dataScaled[step - time_sequence_length: step],
                                                         time_sequence_length)
        predictData.append(model.predict(X_train)[0][0])
    return predictData


# arima
def new_arima(train_data):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    def determine_arima_parameters(train_data):
        # 进行ADF检验，直到数据平稳
        diff_order = 0
        while adfuller(train_data)[1] > 0.05:
            train_data = np.diff(train_data)
            diff_order += 1

        # 绘制ACF和PACF图
        plot_acf(train_data)
        plot_pacf(train_data)

        # 根据ACF和PACF图选择合适的p和q值
        # 这里可以根据需要设置合适的范围来搜索最佳的参数
        p_values = range(0, 2)
        q_values = range(0, 2)
        best_aic = float('inf')
        best_params = None

        for p in p_values:
            for q in q_values:
                try:
                    model = ARIMA(train_data, order=(p, diff_order, q))
                    results = model.fit()
                    aic = results.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, diff_order, q)
                except:
                    continue

        return best_params

    # 自动确定ARIMA模型的参数
    p, d, q = determine_arima_parameters(train_data)
    # p, d, q = 2,2,2
    # 构建ARIMA模型
    model = ARIMA(train_data, order=(p, d, q))
    # 构建ARIMA模型并拟合
    # model = ARIMA(train_data, order=(1, 1, 0))  # 这里的参数(order)需要根据你的数据进行调整
    model = model.fit()
    predictData = []
    # for step in range(0, testDataLength - trainDataLength - future_steps):
    #     x1 = model.forecast(steps=testDataLength - trainDataLength - future_steps)
    #     x2 = scaler.inverse_transform(np.array([x1]))
    #     predictData.append(x1[0])
    x1 = model.forecast(steps=testDataLength - trainDataLength - future_steps)
    for x in range(validDataLength, len(x1)):
        predictData.append([x1[x]])
    return predictData


# lstm
def new_lstm(dataScaled):
    def gru_prepare_data(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps - future_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps: i + time_steps + future_steps, 0])
        return np.array(X), np.array(y)

    # 构建 LSTM 模型
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(time_sequence_length, 1)))
    model.add(Dense(units=future_steps))

    X_train, y_train = gru_prepare_data(dataScaled[: trainDataLength], time_sequence_length)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = gru_prepare_data(dataScaled[trainDataLength: trainDataLength + validDataLength],
                                    time_sequence_length)

    learning_rate = 0.05
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)  # 可以根据需要调整 clipvalue 的值
    # 定义早停回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_times, restore_best_weights=True)
    # 编译模型
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # 训练模型
    model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False, validation_data=(X_val, y_val),
              callbacks=[early_stopping])
    predictData = []
    for step in range(trainDataLength + validDataLength, testDataLength - future_steps, future_steps):
        X = np.array([dataScaled[step - time_sequence_length: step, 0]])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predictData.append(model.predict(X)[0][0])
    return predictData


def new_double_lstm(dataScaled):
    def gru_prepare_data(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps - future_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps: i + time_steps + future_steps, 0])
        return np.array(X), np.array(y)

    # 构建 LSTM 模型
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(time_sequence_length, 1), return_sequences=True))
    model.add(LSTM(units=units2))
    model.add(Dense(units=future_steps))

    X_train, y_train = gru_prepare_data(dataScaled[: trainDataLength], time_sequence_length)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = gru_prepare_data(dataScaled[trainDataLength: trainDataLength + validDataLength],
                                    time_sequence_length)

    learning_rate = 0.05
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)  # 可以根据需要调整 clipvalue 的值
    # 定义早停回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_times, restore_best_weights=True)
    # 编译模型
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # 训练模型
    model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False, validation_data=(X_val, y_val),
              callbacks=[early_stopping])
    predictData = []
    for step in range(trainDataLength + validDataLength, testDataLength - future_steps, future_steps):
        X = np.array([dataScaled[step - time_sequence_length: step, 0]])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predictData.append(model.predict(X)[0][0])
    return predictData


def new_double_lstm_with_attention(dataScaled):
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

    class ModelCheckpointAtEpoch(Callback):
        def __init__(self, data_scaled, time_sequence_length, future_steps, epochs_to_save, batchSize, units, units2,
                     attention_units):
            # 传入需要的数据
            self.data_scaled = data_scaled
            self.time_sequence_length = time_sequence_length
            self.future_steps = future_steps
            self.epochs_to_save = epochs_to_save
            self.batchSize = batchSize
            self.units = units
            self.units2 = units2
            self.attention_units = attention_units

        def on_epoch_end(self, epoch, logs=None):
            # 检查当前epoch是否在指定的列表中
            if epoch in self.epochs_to_save:
                flag = 0
                if os.path.exists(excel_file):
                    with open(excel_file, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if row[0] == key and int(row[1]) == epoch and int(row[2]) == self.batchSize and int(
                                    row[3]) == self.units and int(row[4]) == self.units2 and int(row[5]) == self.attention_units:
                                flag = 1
                                break
                if flag == 0:
                    predict_data = []
                    # 执行保存操作
                    for step in range(trainDataLength + validDataLength, testDataLength - self.future_steps,
                                      self.future_steps):
                        X = np.array([self.data_scaled[step - self.time_sequence_length: step, 0]])
                        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                        predict_data.append(self.model.predict(X)[0][0])

                    # 你可以在这里加入将 self.predict_data 保存到文件的代码
                    rmse, mse, mae, ar2 = rmse_and_mse_mae_compute(actualData, predict_data,
                                                                   "double_lstm_with_attention")
                    # 将结果添加到 DataFrame 中
                    # 直接写入 csv 文件
                    with open(excel_file, 'a', newline='') as f:
                        csv_write = csv.writer(f)
                        csv_write.writerow(
                            ["double_lstm_with_attention", epoch, self.batchSize, self.units, self.units2,
                             self.attention_units, rmse, mse, mae, ar2])

    def gru_prepare_data(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps - future_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps: i + time_steps + future_steps, 0])
        return np.array(X), np.array(y)

    # 构建 LSTM 模型
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(time_sequence_length, 1), return_sequences=True))
    model.add(LSTM(units=units2, return_sequences=True))
    model.add(AdditiveAttention(attention_units=attention_units))
    model.add(Dense(units=future_steps))

    X_train, y_train = gru_prepare_data(dataScaled[: trainDataLength], time_sequence_length)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = gru_prepare_data(dataScaled[trainDataLength: trainDataLength + validDataLength],
                                    time_sequence_length)

    learning_rate = 0.05
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)  # 可以根据需要调整 clipvalue 的值
    # 定义早停回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_times, restore_best_weights=True)
    # 编译模型
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    epochs_to_save = [500, 1000, 1500, 2000, 2500, 3000, 3500]
    checkpoint_callback = ModelCheckpointAtEpoch(dataScaled, time_sequence_length, future_steps, epochs_to_save,
                                                 batchSize, units, units2, attention_units)
    # 训练模型
    model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False, validation_data=(X_val, y_val),
              callbacks=[early_stopping, checkpoint_callback])
    predictData = []
    for step in range(trainDataLength + validDataLength, testDataLength - future_steps, future_steps):
        X = np.array([dataScaled[step - time_sequence_length: step, 0]])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predictData.append(model.predict(X)[0][0])
    return predictData


# transformer
def new_transformer(dataScaled):
    def gru_prepare_data(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps - future_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps: i + time_steps + future_steps, 0])
        return np.array(X), np.array(y)

    def transformer_model(input_shape, output_shape, num_layers, emb_dim, num_units, activation, loss_function,
                          optimizer, batch_size, epochs):
        inputs = tf.keras.Input(shape=input_shape)  # 定义输入层
        x = inputs
        from tensorflow.keras import layers, Model
        # Transformer Encoder
        for _ in range(num_layers):
            # Multi-head self-attention
            query = x
            key_value = x
            attention_output = layers.MultiHeadAttention(num_heads=1, key_dim=emb_dim)(query, key_value)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

        # Output layer
        outputs = layers.Dense(units=output_shape[1])(x)  # 输出层的单元数应与输出形状的第二维匹配

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(optimizer=optimizer, loss=loss_function)

        return model

    input_shape = (time_sequence_length, future_steps)  # 输入形状
    output_shape = (1, 1)  # 输出形状
    num_layers = 1
    emb_dim = 32
    num_units = 0
    activation = 'Sigmoid'
    loss_function = 'mean_squared_error'
    optimizer = tf.keras.optimizers.Adam()
    batch_size = 512
    epochs = 1000

    # 创建并训练模型
    model = transformer_model(input_shape=input_shape, output_shape=output_shape, num_layers=num_layers,
                              emb_dim=emb_dim,
                              num_units=num_units, activation=activation, loss_function=loss_function,
                              optimizer=optimizer, batch_size=batch_size, epochs=epochs)

    # # 模型参数
    # time_steps = time_sequence_length
    # d_model = 128
    # num_heads = 1
    # num_layers = 1
    # dropout = 0.001
    # inputs = Input(shape=(time_steps, 1))
    # x = inputs
    # for _ in range(num_layers):
    #     x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    #     x = Dropout(dropout)(x)
    #     x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    #     y = Dense(units=d_model, activation='linear')(x)
    #     y = Dense(units=1)(y)
    #     x = tf.keras.layers.Add()([x, y])
    #     x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # outputs = Dense(units=1)(x)
    # model = Model(inputs=inputs, outputs=outputs)

    X_train, y_train = gru_prepare_data(dataScaled[: trainDataLength], time_sequence_length)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = gru_prepare_data(dataScaled[trainDataLength: trainDataLength + validDataLength],
                                    time_sequence_length)

    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)  # 可以根据需要调整 clipvalue 的值
    # 定义早停回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_times, restore_best_weights=True)
    # 编译模型
    # model.compile(optimizer=optimizer, loss='mean_squared_error')
    # 训练模型
    model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False, validation_data=(X_val, y_val),
              callbacks=[early_stopping])

    predictData = []
    for step in range(trainDataLength + validDataLength, testDataLength - future_steps, future_steps):
        X = np.array([dataScaled[step - time_sequence_length: step, 0]])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predictData.append(model.predict(X)[0][0])
    return predictData


# sma
def new_sma(dataScaled):
    predictData = []
    windows = 2
    for step in range(trainDataLength + validDataLength, testDataLength - future_steps, future_steps):
        result = 0
        for n in range(1, 1 + windows):
            result += dataScaled[step - n][0]
        predictData.append(result / windows)
    return predictData


# gru
def new_gru(dataScaled):
    def gru_prepare_data(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps - future_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps: i + time_steps + future_steps, 0])
        return np.array(X), np.array(y)

    np.random.seed(42)
    tf.random.set_seed(42)
    # 模型参数
    input_dim = features  # 输入数据的特征维度
    gru_units = units  # GRU单元数量
    gru_units2 = units2  # GRU单元数量
    output_dim = future_steps  # 输出维度
    # 创建GRU模型，并添加注意力层
    model = Sequential([
        GRU(gru_units, input_shape=(time_sequence_length, input_dim)),
        Dense(output_dim, kernel_regularizer=l2(l2_regularization))
    ])
    X_train, y_train = gru_prepare_data(dataScaled[: trainDataLength], time_sequence_length)
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_val, y_val = gru_prepare_data(dataScaled[trainDataLength:
                                               trainDataLength + validDataLength], time_sequence_length)
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)  # 可以根据需要调整 clipvalue 的值
    # 定义早停回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_times,
                                   restore_best_weights=True)
    # 编译模型
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # 训练模型
    model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False,
              validation_data=(X_val, y_val), callbacks=[early_stopping])
    # callbacks=[early_stopping]
    # 保存模型权重
    model.save_weights('model_weights.h5')
    # 预测值
    predictData = []
    for step in range(trainDataLength + validDataLength, testDataLength - future_steps, future_steps):
        X = np.array([dataScaled[step - time_sequence_length: step, 0]])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        tmpPredict = model.predict(X)[0]
        predictData.append(tmpPredict)
    return predictData


def new_double_gru(dataScaled):
    def gru_prepare_data(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps - future_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps: i + time_steps + future_steps, 0])
        return np.array(X), np.array(y)

    np.random.seed(42)
    tf.random.set_seed(42)
    # 修改注意力层的参数
    attention_units = 1  # 增加注意力权重的维度
    # 模型参数
    input_dim = features  # 输入数据的特征维度
    gru_units = units  # GRU单元数量
    gru_units2 = units2  # GRU单元数量
    output_dim = future_steps  # 输出维度
    # 创建GRU模型，并添加注意力层
    model = Sequential([
        GRU(gru_units, input_shape=(time_sequence_length, input_dim), return_sequences=True),
        GRU(gru_units2),
        Dense(output_dim, kernel_regularizer=l2(l2_regularization))
    ])
    X_train, y_train = gru_prepare_data(dataScaled[: trainDataLength], time_sequence_length)
    X_train, y_train = np.array(X_train), np.array(y_train)
    print(len(X_train))
    X_val, y_val = gru_prepare_data(dataScaled[trainDataLength:
                                               trainDataLength + validDataLength], time_sequence_length)
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)  # 可以根据需要调整 clipvalue 的值
    # 定义早停回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_times,
                                   restore_best_weights=True)
    # 编译模型
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # 训练模型
    model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False,
              validation_data=(X_val, y_val), callbacks=[early_stopping])
    # model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False,
    #           validation_data=(X_val, y_val))

    # 保存模型权重
    model.save_weights('model_weights.h5')
    # 预测值
    predictData = []
    for step in range(trainDataLength + validDataLength, testDataLength - future_steps, future_steps):
        X = np.array([dataScaled[step - time_sequence_length: step, 0]])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        tmpPredict = model.predict(X)[0]
        predictData.append(tmpPredict)
    return predictData


# gru_with_residual
def new_gru_with_residual(dataScaled):
    def gru_prepare_data(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps - future_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps: i + time_steps + future_steps, 0])
        return np.array(X), np.array(y)

    np.random.seed(42)
    tf.random.set_seed(42)
    # 修改注意力层的参数
    attention_units = 1  # 增加注意力权重的维度
    # 模型参数
    input_dim = features  # 输入数据的特征维度
    gru_units = units  # GRU单元数量
    gru_units2 = units2  # GRU单元数量
    output_dim = future_steps  # 输出维度
    # 创建GRU模型，并添加注意力层
    model = Sequential([
        GRU(gru_units, input_shape=(time_sequence_length, input_dim), return_sequences=True),
        GRU(gru_units2),
        Dense(output_dim, kernel_regularizer=l2(l2_regularization))
    ])
    X_train, y_train = gru_prepare_data(dataScaled[: trainDataLength], time_sequence_length)
    X_train, y_train = np.array(X_train), np.array(y_train)
    print(len(X_train))
    X_val, y_val = gru_prepare_data(dataScaled[trainDataLength:
                                               trainDataLength + validDataLength], time_sequence_length)
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)  # 可以根据需要调整 clipvalue 的值
    # 定义早停回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_times,
                                   restore_best_weights=True)
    # 编译模型
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # 训练模型
    model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False,
              validation_data=(X_val, y_val), callbacks=[early_stopping])
    # model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False,
    #           validation_data=(X_val, y_val))

    # 保存模型权重
    model.save_weights('model_weights.h5')
    # 预测值
    predictData = []
    index = 0
    # 先预测过去residualLength个时间间隔的数据
    for step in range(trainDataLength + validDataLength - residualLength * future_steps,
                      trainDataLength + validDataLength, future_steps):
        X = np.array([dataScaled[step - time_sequence_length: step, 0]])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predictData.append(model.predict(X)[0])
    for step in range(trainDataLength + validDataLength, testDataLength - future_steps, future_steps):
        X = np.array([dataScaled[step - time_sequence_length: step, 0]])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        residual = {}
        tmpPredict = model.predict(X)[0]
        for n in range(1, 1 + residualLength):
            predict = predictData[len(predictData) - n]
            for stepLen in range(0, len(tmpPredict)):
                if stepLen in residual:
                    residual[stepLen] += actualDataWithResidual[index + residualLength - n][stepLen] - predict[stepLen]
                else:
                    residual[stepLen] = actualDataWithResidual[index + residualLength - n][stepLen] - predict[stepLen]
        for stepLen in range(0, len(tmpPredict)):
            tmpPredict[stepLen] += residual[stepLen] / residualLength
        predictData.append(tmpPredict)
        index += 1
    return predictData[residualLength:]


# gru_with_attention
def new_gru_with_attention(dataScaled):
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

    def gru_prepare_data(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps - future_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps: i + time_steps + future_steps, 0])
        return np.array(X), np.array(y)

    np.random.seed(42)
    tf.random.set_seed(42)
    # 修改注意力层的参数
    # attention_units = 1  # 增加注意力权重的维度
    # 模型参数
    input_dim = features  # 输入数据的特征维度
    gru_units = units  # GRU单元数量
    gru_units2 = units2  # GRU单元数量
    output_dim = future_steps  # 输出维度
    # 创建GRU模型，并添加注意力层
    model = Sequential([
        GRU(gru_units, input_shape=(time_sequence_length, input_dim), return_sequences=True),
        GRU(gru_units2, return_sequences=True),
        AdditiveAttention(attention_units),
        Dense(output_dim, kernel_regularizer=l2(l2_regularization))
    ])

    X_train, y_train = gru_prepare_data(dataScaled[: trainDataLength], time_sequence_length)
    X_train, y_train = np.array(X_train), np.array(y_train)
    print(len(X_train))
    X_val, y_val = gru_prepare_data(dataScaled[trainDataLength:
                                               trainDataLength + validDataLength], time_sequence_length)
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)  # 可以根据需要调整 clipvalue 的值
    # 定义早停回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_times,
                                   restore_best_weights=True)
    # 编译模型
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # 训练模型
    model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False,
              validation_data=(X_val, y_val), callbacks=[early_stopping])
    # model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False,
    #           validation_data=(X_val, y_val))

    # 保存模型权重
    model.save_weights('model_weights.h5')
    # 预测值
    predictData = []
    for step in range(trainDataLength + validDataLength, testDataLength - future_steps, future_steps):
        X = np.array([dataScaled[step - time_sequence_length: step, 0]])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        tmpPredict = model.predict(X)[0]
        predictData.append(tmpPredict)
    return predictData


# my_gru
def new_my_gru(dataScaled):
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

    def gru_prepare_data(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps - future_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps: i + time_steps + future_steps, 0])
        return np.array(X), np.array(y)

    np.random.seed(42)
    tf.random.set_seed(42)
    # 模型参数
    input_dim = features  # 输入数据的特征维度
    gru_units = units  # GRU单元数量
    gru_units2 = units2  # GRU单元数量
    output_dim = future_steps  # 输出维度
    # 创建GRU模型，并添加注意力层
    model = Sequential([
        GRU(gru_units, input_shape=(time_sequence_length, input_dim), return_sequences=True),
        GRU(gru_units2, return_sequences=True),
        AdditiveAttention(attention_units),
        Dense(output_dim, kernel_regularizer=l2(l2_regularization))
    ])

    X_train, y_train = gru_prepare_data(dataScaled[: trainDataLength], time_sequence_length)
    X_train, y_train = np.array(X_train), np.array(y_train)
    print(len(X_train))
    X_val, y_val = gru_prepare_data(dataScaled[trainDataLength:
                                               trainDataLength + validDataLength], time_sequence_length)
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)  # 可以根据需要调整 clipvalue 的值
    # 定义早停回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_times,
                                   restore_best_weights=True)
    # 编译模型
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # 训练模型
    model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False,
              validation_data=(X_val, y_val), callbacks=[early_stopping])
    # model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=False,
    #           validation_data=(X_val, y_val))

    # 保存模型权重
    model.save_weights('model_weights.h5')
    # model.save(f'workloadPredictA.h5')
    # 预测值
    predictData = []
    index = 0
    # 先预测过去residualLength个时间间隔的数据
    for step in range(trainDataLength + validDataLength - residualLength * future_steps,
                      trainDataLength + validDataLength, future_steps):
        X = np.array([dataScaled[step - time_sequence_length: step, 0]])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predictData.append(model.predict(X)[0])
    for step in range(trainDataLength + validDataLength, testDataLength - future_steps, future_steps):
        X = np.array([dataScaled[step - time_sequence_length: step, 0]])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        residual = {}
        tmpPredict = model.predict(X)[0]
        for n in range(1, 1 + residualLength):
            predict = predictData[len(predictData) - n]
            for stepLen in range(0, len(tmpPredict)):
                if stepLen in residual:
                    # if abs(actualDataWithResidual[index + residualLength - n][stepLen] - predict[stepLen])/actualDataWithResidual[index + residualLength - n][stepLen] < 0.05:
                    #     residual[stepLen] += actualDataWithResidual[index + residualLength - n][stepLen] - predict[stepLen]
                    # else:
                    #     residual[stepLen] += 0
                    residual[stepLen] += actualDataWithResidual[index + residualLength - n][stepLen] - predict[stepLen]
                else:
                    # if abs(actualDataWithResidual[index + residualLength - n][stepLen] - predict[stepLen])/actualDataWithResidual[index + residualLength - n][stepLen] < 0.05:
                    #     residual[stepLen] = actualDataWithResidual[index + residualLength - n][stepLen] - predict[stepLen]
                    # else:
                    #     residual[stepLen] = 0
                    residual[stepLen] = actualDataWithResidual[index + residualLength - n][stepLen] - predict[stepLen]

        for stepLen in range(0, len(tmpPredict)):
            if residualLength != 0:
                tmpPredict[stepLen] += residual[stepLen] / residualLength
        predictData.append(tmpPredict)
        index += 1
    return predictData[residualLength:]


if __name__ == "__main__":
    # 获取当前模块的名称
    current_module = importlib.import_module('__main__')
    # 数据处理
    np.random.seed(42)
    tf.random.set_seed(42)
    # workload.xlsx
    # scaled_requests.xlsx df.values[5100: ]
    df = pd.read_excel(filePath, header=None, engine='openpyxl')[1]

    # 归一化数据 输入归一化，输出不需要归一化
    data = df.values
    scaler = MinMaxScaler()
    # 原始值
    dataNoScaled = np.array(data).reshape(-1, 1)
    # 归一化数据
    dataScaled = scaler.fit_transform(dataNoScaled)
    # print(dataScaled)

    # ARIMA所需要的数据
    trainData = dataScaled[: trainDataLength]

    # 需要用到结果的测试集
    originActualData = dataScaled[trainDataLength + validDataLength: testDataLength]
    actualData = []
    for i in range(0, len(originActualData) - future_steps, future_steps):
        tmpData = []
        for step in range(i, i + future_steps):
            tmpData.append(originActualData[step][0])
        actualData.append(tmpData)
    # 总的测试集
    originActualDataWithResidual = dataScaled[trainDataLength + validDataLength -
                                              residualLength: testDataLength]
    actualDataWithResidual = []
    for i in range(0, len(originActualDataWithResidual) - future_steps):
        tmpData = []
        for step in range(i, i + future_steps):
            tmpData.append(originActualDataWithResidual[step][0])
        actualDataWithResidual.append(tmpData)
    print(f"actual: {len(actualData)}")
    print(f"with residual: {len(actualDataWithResidual)}")

    # 原始值
    # 需要用到结果的测试集
    originActualDataNoScaled = dataNoScaled[trainDataLength + validDataLength: testDataLength]
    actualDataNoScaled = []
    for i in range(0, len(originActualDataNoScaled) - future_steps, future_steps):
        tmpData = []
        for step in range(i, i + future_steps):
            tmpData.append(originActualDataNoScaled[step][0])
        actualDataNoScaled.append(tmpData)
    # 总的测试集
    originActualDataWithResidualNoScaled = dataNoScaled[trainDataLength + validDataLength -
                                                        residualLength: testDataLength]
    actualDataWithResidualNoScaled = []
    for i in range(0, len(originActualDataWithResidualNoScaled) - future_steps):
        tmpData = []
        for step in range(i, i + future_steps):
            tmpData.append(originActualDataWithResidualNoScaled[step][0])
        actualDataWithResidualNoScaled.append(tmpData)
    print(f"actual: {len(actualDataNoScaled)}")
    print(f"with residual: {len(actualDataWithResidualNoScaled)}")

    modelNames = {"linearregression": [], "arima": [], "lstm": [], "transformer": [], "sma": [], "gru": [],
                  "double_gru": [], "gru_with_residual": [],
                  "gru_with_attention": [], "my_gru": []}
    modelParameters = {"linearregression": dataScaled, "arima": trainData, "transformer": dataScaled,
                       "lstm": dataScaled, "sma": dataScaled
        , "gru": dataScaled, "double_gru": dataScaled, "gru_with_residual": dataScaled,
                       "gru_with_attention": dataScaled,
                       "my_gru": dataScaled}
    modelResults = {"linearregression": [], "arima": [], "lstm": [], "transformer": [], "sma": [], "gru": [],
                    "double_gru": [], "gru_with_residual": [],
                    "gru_with_attention": [], "my_gru": []}

    # modelNames = {"arima": [], "lstm": [], "sma": [], "gru": [], "double_gru": [], "gru_with_residual": [],
    #               "gru_with_attention": [], "my_gru": []}
    # modelParameters = {"arima": trainData, "lstm": dataScaled, "sma": dataScaled
    #                    , "gru": dataScaled, "double_gru": dataScaled, "gru_with_residual": dataScaled, "gru_with_attention": dataScaled,
    #                    "my_gru": dataScaled}
    # modelResults = {"arima": [], "lstm": [], "sma": [], "gru": [], "double_gru": [], "gru_with_residual": [],
    #                 "gru_with_attention": [], "my_gru": []}

    # modelNames = {"linearregression": [], "arima": [], "lstm": [], "transformer": [], "sma": [], "gru": [], "gru_with_residual": [],
    #               "gru_with_attention": [], "my_gru": []}
    # modelParameters = {"linearregression": dataScaled, "arima": trainData, "lstm": dataScaled, "transformer": dataScaled, "sma": dataScaled
    #                    , "gru": dataScaled, "gru_with_residual": dataScaled, "gru_with_attention": dataScaled,
    #                    "my_gru": dataScaled}
    # modelResults = {"linearregression": [], "arima": [], "lstm": [], "transformer": [], "sma": [], "gru": [], "gru_with_residual": [],
    #                 "gru_with_attention": [], "my_gru": []}

    # modelNames = {"transformer": []}
    # modelParameters = {"transformer": dataScaled}
    # modelResults = { "transformer": []}

    # modelNames = {"my_gru": [],
    #               "gru_with_attention": []}
    # modelParameters = {"my_gru": dataScaled, "gru_with_attention": dataScaled}
    # modelResults = { "my_gru": [],
    #                 "gru_with_attention": []}

    # modelNames = {
    #               "gru_with_attention": []}
    # modelParameters = { "gru_with_attention": dataScaled}
    # modelResults = {
    #                 "gru_with_attention": []}

    # modelNames = {"my_gru": [], "double_gru": []}
    # modelParameters = {"my_gru": dataScaled, "double_gru": dataScaled}
    # modelResults = {"my_gru": [], "my_gru": []}
    # modelNames = {"my_gru": [], "gru": []}
    # modelParameters = {"my_gru": dataScaled, "gru": dataScaled}
    # modelResults = {"my_gru": [], "gru": []}

    # modelNames = {"gru": []}
    # modelParameters = {"gru": dataScaled}
    # modelResults = {"gru": []}

    # modelNames = {"my_gru": []}
    # modelParameters = {"my_gru": dataScaled}
    # modelResults = {"my_gru": []}

    # modelNames = {"lstm": []}
    # modelParameters = {"lstm": dataScaled}
    # modelResults = {"lstm": []}

    # modelNames = {"double_lstm": []}
    # modelParameters = {"double_lstm": dataScaled}
    # modelResults = {"double_lstm": []}

    modelNames = {"double_lstm_with_attention": []}
    modelParameters = {"double_lstm_with_attention": dataScaled}
    modelResults = {"double_lstm_with_attention": []}

    # modelNames = {"double_lstm_with_mul_attention": []}
    # modelParameters = {"double_lstm_with_mul_attention": dataScaled}
    # modelResults = {"double_lstm_with_mul_attention": []}

    # modelNames = {"linearregression": []}
    # modelParameters = {"linearregression": dataScaled}
    # modelResults = {"linearregression": []}

    # modelNames = {"linearregression": [], "my_gru": []}
    # modelParameters = {"linearregression": dataScaled, "my_gru": dataScaled}
    # modelResults = {"linearregression": [], "my_gru": []}

    #
    # modelNames = {"gru": [], "double_gru": []}
    # modelParameters = {"gru": dataScaled, "double_gru": dataScaled}
    # modelResults = {"gru": [], "double_gru": []}
    #
    # modelNames = {"double_gru": []}
    # modelParameters = {"double_gru": dataScaled}
    # modelResults = {"double_gru": []}

    # modelNames = {"my_gru": [],
    #               "gru_with_residual": []}
    # modelParameters = {"my_gru": dataScaled, "gru_with_residual": dataScaled}
    # modelResults = { "my_gru": [], "gru_with_residual": []}

    # modelNames = {"my_gru": [],
    #               "gru_with_residual": [], "gru_with_attention": []}
    # modelParameters = {"my_gru": dataScaled, "gru_with_residual": dataScaled,
    #                    "gru_with_attention": dataScaled}
    # modelResults = { "my_gru": [], "gru_with_residual": [], "gru_with_attention": []}

    # modelNames = {"arima": []}
    # modelParameters = {"arima": dataScaled}
    # modelResults = {"arima": []}

    # modelNames = { "gru": [], "double_gru": [], "gru_with_residual": [],
    #               "gru_with_attention": [], "my_gru": []}
    # modelParameters = {"gru": dataScaled, "double_gru": dataScaled, "gru_with_residual": dataScaled, "gru_with_attention": dataScaled,
    #                    "my_gru": dataScaled}
    # modelResults = {"gru": [], "double_gru": [], "gru_with_residual": [],
    #                 "gru_with_attention": [], "my_gru": []}

    # results_df = pd.DataFrame(columns=['epochs', 'batchsize', 'units1', 'units2', "RMSE", "MSE", "MAE", "AR2"])

    # result = []
    # excel_file = 'results.csv'
    # for key in modelNames.keys():
    #     modelNames[key] = getattr(current_module, f"new_{key}")(modelParameters[key])
    # print(f"数据集: {filePath}")
    # for key in modelNames.keys():
    #     modelResults[key] = rmse_and_mse_mae_compute(actualData, modelNames[key], key)
    #     rmse, mse, mae, ar2 = modelResults[key]
    #     result.append([key, mse, rmse, ar2, mae])
    #     # results_df = results_df.append({'epochs': epochs, 'batchsize': batchSize, 'units1': units, 'units2': units2,
    #     #                                 'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'AR2': ar2}, ignore_index=True)
    # # results_df.to_excel(excel_file, index=False)
    # df = pd.DataFrame(result, columns=['model', 'MSE', 'RMSE', 'AR2', 'MAE'])
    # # 保存到csv文件
    # df.to_csv(excel_file, index=False)

    # 随机搜索
    # param_space = {'batch_size': [16, 32, 64, 128],
    #         #    'epoch': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    #                'epoch': [300, 500],
    #            'opt': [Adam(lr=0.001), Adam(lr=0.01), Adam(lr=0.1)],
    #            "units": [2, 4, 8, 16, 32, 64, 128],
    #            "units2": [2, 4, 8, 16, 32, 64, 128]}
    # params_list = list(ParameterSampler(param_space, n_iter=10, random_state=42))
    # mse_list = []
    # results_df = pd.DataFrame(columns=['epochs', 'batchsize', 'units1', 'units2', "RMSE", "MSE", "MAE", "AR2"])
    # excel_file = 'results.xlsx'
    # for params in params_list:
    #     epochs, batchSize, units, units2 = params['batch_size'], params['epoch'], params["units"], params["units2"]
    #     for key in modelNames.keys():
    #         modelNames[key] = getattr(current_module, f"new_{key}")(modelParameters[key])
    #         print(f"数据集: {filePath}")
    #     results_df = pd.DataFrame(columns=['epochs', 'batchsize', 'units1', 'units2', "RMSE", "MSE", "MAE", "AR2"])
    #     for key in modelNames.keys():
    #         modelResults[key] = rmse_and_mse_mae_compute(actualData, modelNames[key], key)
    #                                         # 将结果添加到 DataFrame 中
    #         mse_list.append(modelResults[key][1])
    #         rmse, mse, mae, ar2 = modelResults[key]
    #         results_df = results_df.append({'epochs': epochs, 'batchsize': batchSize, 'units1': units, 'units2': units2,
    #                                             'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'AR2': ar2}, ignore_index=True)
    #         results_df.to_excel(excel_file, index=False)

    # best_params = params_list[np.argmin(mse_list)]
    # print('best params:', best_params)
    # print('best MSE:', np.min(mse_list))

    # 网格搜索
    excel_file = 'results.csv'
    # csv文件头
    with open(excel_file, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(['model', 'epochs', 'batchsize', 'units1', 'units2', 'units3', 'MSE', 'RMSE', 'MAE', 'AR2'])
    epochs = 3500
    # epochList = [500, 1000, 1500, 2000, 2500, 3000, 3500]
    # batchSize = 128
    batchList = [64, 128, 256, 512, 1024]
    # units = 16
    # units2 = 32
    unitList = [4, 8, 16, 32, 64]
    # results_df = []
    # for a in epochList:
    # epochs = epochs
    for b in batchList:
        batchSize = b
        for c in unitList:
            units = c
            for d in unitList:
                units2 = d
                for e in unitList:
                    attention_units = e
                    # for key in modelNames.keys():
                    for key in modelNames.keys():
                        # 判断文件是否存在
                        # 判断csv文件中是否有相同key, epochs, batchSize, units, units2的行
                        # 如果有则不写入
                        flag = 0
                        if os.path.exists(excel_file):
                            with open(excel_file, 'r') as f:
                                reader = csv.reader(f)
                                for row in reader:
                                    if row[0] == key and int(row[1]) == epochs and int(row[2]) == batchSize and int(
                                            row[3]) == units and int(row[4]) == units2 and int(row[5]) == attention_units:
                                        flag = 1
                                        break
                        if flag == 0:
                            print(f"数据集: {filePath}")
                            print(
                                f"开始训练：\nepochs: {epochs}, batchSize: {batchSize}, units: {units}, units2: {units2}, attention_units: {attention_units}")
                            modelNames[key] = getattr(current_module, f"new_{key}")(modelParameters[key])
                            modelResults[key] = rmse_and_mse_mae_compute(actualData, modelNames[key], key)
                            # 将结果添加到 DataFrame 中
                            rmse, mse, mae, ar2 = modelResults[key]
                            # 直接写入 csv 文件
                            with open(excel_file, 'a', newline='') as f:
                                csv_write = csv.writer(f)
                                csv_write.writerow([key, epochs, batchSize, units, units2, mse, rmse, mae, ar2])

                            # results_df.append([key, epochs, batchSize, units, units2, mse, rmse, mae, ar2])

    # df = pd.DataFrame(results_df,
    #                   columns=['model', 'epochs', 'batchsize', 'units1', 'units2', 'MSE', 'RMSE', 'MAE', 'AR2'])
    # # 将结果保存到 Excel 文件中
    # df.to_csv(excel_file, index=False)

    # units = 2
    # units2 = 2
    # attention_units = 2
    # unitList = [4, 8, 16]
    # results = []
    # result2 = []
    # for c in unitList:
    #     units = c
    #     for d in unitList:
    #         units2 = d
    #         for e in unitList:
    #             attention_units = e
    #             for key in modelNames.keys():
    #                     modelNames[key] = getattr(current_module, f"new_{key}")(modelParameters[key])
    #                     print(f"数据集: {filePath}")
    #                     for key in modelNames.keys():
    #                         modelResults[key] = rmse_and_mse_mae_compute(actualData, modelNames[key], key)
    #                                             # 将结果添加到 DataFrame 中
    #                         rmse, mse, mae, ar2 = modelResults[key]
    #                         results.append([epochs, batchSize, units, units2,
    #                                         attention_units, mse, rmse, ar2, mae])
    #                         result2.append(f"units:{units}\nunits2:{units2}\nattention_units:{attention_units}\nmse:{mse}\nrmse:{rmse}\nar2:{ar2}\nmae:{mae}")

    # print(results)
    # df = pd.DataFrame(results)
    # df.to_excel(f"result.xlsx", index=False, header=None)

    # batchSize = 32
    # batchSizeList = [64, 128, 256, 512, 1024]
    # results = []
    # result2 = []
    # for b in batchSizeList:
    #     batchSize = b
    #     for key in modelNames.keys():
    #         modelNames[key] = getattr(current_module, f"new_{key}")(modelParameters[key])
    #         print(f"数据集: {filePath}")
    #         for key in modelNames.keys():
    #             modelResults[key] = rmse_and_mse_mae_compute(actualData, modelNames[key], key)
    #             # 将结果添加到 DataFrame 中
    #             rmse, mse, mae, ar2 = modelResults[key]
    #             results.append([epochs, batchSize, units, units2,
    #                             attention_units, mse, rmse, ar2, mae])
    #             result2.append(f"units:{units}\nunits2:{units2}\nattention_units:{attention_units}\nmse:{mse}\nrmse:{rmse}\nar2:{ar2}\nmae:{mae}")
    # print(results)
    # print(result2)
    # units = 2
    # units2 = 2
    # attention_units = 2
    # unitsList = [[4,8,8],[4,8,16],[8,4,4],[8,4,8],
    #          [8,4,16],[8,8,4],[8,8,8],[8,8,16],
    #          [8,16,4],[8,16,8],[8,16,16],[16,16,16]]
    # unitsList = [[8,16,32]]
    # results = []
    # result2 = []
    # for b in unitsList:
    #     units, units2, attention_units = b[0], b[1], b[2]
    #     for key in modelNames.keys():
    #         modelNames[key] = getattr(current_module, f"new_{key}")(modelParameters[key])
    #         print(f"数据集: {filePath}")
    #         for key in modelNames.keys():
    #             modelResults[key] = rmse_and_mse_mae_compute(actualData, modelNames[key], key)
    #             # 将结果添加到 DataFrame 中
    #             rmse, mse, mae, ar2 = modelResults[key]
    #             results.append([epochs, batchSize, units, units2,
    #                             attention_units, mse, rmse, ar2, mae])
    #             result2.append(f"units:{units}\nunits2:{units2}\nattention_units:{attention_units}\nmse:{mse}\nrmse:{rmse}\nar2:{ar2}\nmae:{mae}")
    # print(results)
    # print(result2)

    # df = pd.DataFrame(results)
    # df.to_excel(f"result.xlsx", index=False, header=None)
    # results = []
    # result2 = []
    # epochList = [500]
    # for c in epochList:
    #     epochs = c
    #     for key in modelNames.keys():
    #         modelNames[key] = getattr(current_module, f"new_{key}")(modelParameters[key])
    #         print(f"数据集: {filePath}")
    #         for key in modelNames.keys():
    #             modelResults[key] = rmse_and_mse_mae_compute(actualData, modelNames[key], key)
    #                                 # 将结果添加到 DataFrame 中
    #             rmse, mse, mae, ar2 = modelResults[key]
    #             results.append([epochs, batchSize, units, units2,
    #                             attention_units, mse, rmse, ar2, mae])
    #             result2.append(f"units:{units}\nunits2:{units2}\nattention_units:{attention_units}\nmse:{mse}\nrmse:{rmse}\nar2:{ar2}\nmae:{mae}")
    # print(results)

    # results = []
    # result2 = []
    # result3 = []
    # epochList = [500]
    # for c in epochList:
    #     epochs = c
    #     for key in modelNames.keys():
    #         modelNames[key] = getattr(current_module, f"new_{key}")(modelParameters[key])
    #         print(f"数据集: {filePath}")
    #         for key in modelNames.keys():
    #             modelResults[key] = rmse_and_mse_mae_compute(actualData, modelNames[key], key)
    #                                 # 将结果添加到 DataFrame 中
    #             rmse, mse, mae, ar2 = modelResults[key]
    #             results.append([epochs, batchSize, units, units2,
    #                             attention_units, mse, rmse, ar2, mae])
    #             result2.append(f"units:{units}\nunits2:{units2}\nattention_units:{attention_units}\nmse:{mse}\nrmse:{rmse}\nar2:{ar2}\nmae:{mae}")
    #             result3.append([time_sequence_length, future_steps])
    # print(results)
    # print(result3)
    # print("------------")
