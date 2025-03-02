import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, \
    accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
def transform_features(X):
    """
    对输入的时序数据进行特征提取。
    输入 X 的形状为 (n_samples, n_features, n_timesteps)。
    对于每个样本中的每个特征（即每一行），计算4个统计量：
      - 均值（mean）
      - 标准差（std）
      - 最小值（min）
      - 最大值（max）

    然后按顺序将所有特征的各统计量拼接，形成新的特征向量，形状为 (n_samples, n_features * 4)。
    """
    means = np.mean(X, axis=2)
    stds = np.std(X, axis=2)
    mins = np.min(X, axis=2)
    maxs = np.max(X, axis=2)

    return np.concatenate([means, stds, mins, maxs], axis=1)

def flatten_features(X):
    """
    将输入的时序数据展平。
    输入 X 的形状为 (n_samples, n_features, n_timesteps)。
    将每个样本的特征和时间步展平为 (n_samples, n_features * n_timesteps)。
    """
    return X.reshape(X.shape[0], -1)  # 展平为 (n_samples, n_features * n_timesteps)



mse_scores = []
mae_scores = []

# 五折交叉验证
for fold in range(1, 6):
    print(f"\nProcessing fold {fold}/5")


    data_path = f'./data/5_fold_c_f_p_reg/huigui_5_fold_{fold}.pt'
    data = torch.load(data_path)

    # 原始数据：原来 shape 为 (n_samples, time_steps, n_features)，转置后变为 (n_samples, n_features, time_steps)
    x_train = data['train_X'].numpy().transpose(0, 2, 1)
    x_test = data['test_X'].numpy().transpose(0, 2, 1)
    y_train = data['train_y'].numpy()
    y_test = data['test_y'].numpy()


    X_train_transformed = np.concatenate([flatten_features(x_train), transform_features(x_train)], axis=1)
    X_test_transformed = np.concatenate([flatten_features(x_test), transform_features(x_test)], axis=1)
    clf = RandomForestRegressor(n_estimators=200, n_jobs=-1)  # 200

#以下是对比方法

    # from sklearn.svm import SVR
    # # 创建 SVR 模型
    # clf = SVR(kernel='rbf')


    # import xgboost as xgb
    # clf= xgb.XGBRegressor(n_estimators=100)

    # from sklearn.ensemble import AdaBoostRegressor
    # clf = AdaBoostRegressor(n_estimators=100)

    # # 训练随机森林分类器，设定 200 棵树，n_jobs=-1 加速

    clf.fit(X_train_transformed, y_train)
    # 对测试集进行预测
    y_pred = clf.predict(X_test_transformed)
# 计算 MSE 和 MAE
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # 保存当前折的结果
    mse_scores.append(mse)
    mae_scores.append(mae)
    # 打印当前折结果
    print(f"Fold {fold} Results:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
# 计算平均结果和标准差
avg_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
avg_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)
# 输出最终平均结果和标准差
print("\nFinal Average Results with Standard Deviation:")
print(f"Average MSE: {avg_mse:.4f} ± {std_mse:.4f}")
print(f"Average MAE: {avg_mae:.4f} ± {std_mae:.4f}")


