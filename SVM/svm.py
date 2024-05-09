import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import util.osUtil as osUtil

# random = 'random'
random = 'random_random'
# 加载数据
data = osUtil.read_data(random)  # 替换为正确的文件路径
data.drop('job_name', axis=1, inplace=True)
#
data = pd.get_dummies(data, columns=['task_name', 'gpu_type'])

# 选择特征和标签
X = data.drop(['task_type'], axis=1)
y = data['task_type']

# 对标签进行数值编码
y_encoded, _ = pd.factorize(y)  # 将分类标签转换为数值

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=0.8, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建并训练SVM模型
svm_model = SVC(kernel='rbf', C=1.0, gamma='auto', verbose=True)
svm_model.fit(X_train_scaled, y_train)

# 进行预测
y_pred = svm_model.predict(X_test_scaled)

# 分别计算训练集和测试集的准确率
train_accuracy = accuracy_score(y_train, svm_model.predict(X_train_scaled))
test_accuracy = accuracy_score(y_test, y_pred)
print(f'训练集准确率：{train_accuracy}')
print(f'测试集准确率：{test_accuracy}')
