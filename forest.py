import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 加载数据
FILTER_DATA_DIR = 'data/'
data = pd.read_csv(FILTER_DATA_DIR + 'dft.csv')

# 数据预处理
# 填充缺失值，将分类数据转换成数值型数据等
# 在这个例子中，我们需要处理缺失值并且将分类特征转化为dummy变量
data['plan_gpu'].fillna(0, inplace=True)  # 假设没有指定gpu的plan_gpu值可以设置为0
data.fillna('Nan', inplace=True)  # 假设缺失的gpu_type可以用'Nan'填充
data = pd.get_dummies(data, columns=['job_name', 'task_name', 'status'])

# 特征选择
# 我们将使用除了gpu_type之外的所有列作为特征
X = data.drop('gpu_type', axis=1)
y = data['gpu_type']

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林分类器实例
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_clf.fit(X_train, y_train)

# 预测测试集结果
y_pred = rf_clf.predict(X_test)

# 评估模型
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# 如果想保存模型，可以使用joblib
from joblib import dump

dump(rf_clf, 'rf_gpu_type_predictor.joblib')
