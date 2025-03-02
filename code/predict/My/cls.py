import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, \
    accuracy_score,matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV
from imblearn.metrics import geometric_mean_score
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


# 初始化用于保存各折交叉验证结果的列表
roc_auc_scores = []
pr_auc_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
accuracy_scores = []
mcc_scores = []
gmean_scores = []
# 五折交叉验证
for fold in range(1, 6):
    print(f"\nProcessing fold {fold}/5")


    data_path = f'./data/5_fold_c_f_p/final_5_fold_{fold}.pt'

    data = torch.load(data_path)

    x_train = data['train_X'].numpy().transpose(0, 2, 1)
    x_test = data['test_X'].numpy().transpose(0, 2, 1)
    y_train = data['train_y'].numpy()
    y_test = data['test_y'].numpy()


    X_train_transformed = np.concatenate([flatten_features(x_train), transform_features(x_train)], axis=1)
    X_test_transformed = np.concatenate([flatten_features(x_test), transform_features(x_test)], axis=1)

    # # 训练随机森林分类器，设定 200 棵树，n_jobs=-1 加速
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)#200



    # from sklearn.svm import SVC
    # clf = SVC( probability=True)

    # from sklearn.tree import DecisionTreeClassifier
    # # 训练决策树分类器
    # clf = DecisionTreeClassifier()

    import xgboost as xgb
    # # 训练 XGBoost 分类器
    # clf = xgb.XGBClassifier(n_estimators=200, n_jobs=-1)

    # 训练 AdaBoost 分类器
    # from sklearn.ensemble import AdaBoostClassifier
    # clf = AdaBoostClassifier(n_estimators=200)

    clf.fit(X_train_transformed, y_train)
    # 对测试集进行预测
    y_pred = clf.predict(X_test_transformed)
    y_pred_proba = clf.predict_proba(X_test_transformed)

    # 计算各项指标（注意二分类时 ROC_AUC 和 PR_AUC 需要概率预测结果）
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    pr_auc = average_precision_score(y_test, y_pred_proba[:, 1])
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    gmean = geometric_mean_score(y_test, y_pred)
    # 保存当前折的结果
    roc_auc_scores.append(roc_auc)
    pr_auc_scores.append(pr_auc)
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)
    accuracy_scores.append(accuracy)
    mcc_scores.append(mcc)
    gmean_scores.append(gmean)

    # 打印当前折结果
    print(f"Fold {fold} Results:")
    print(f"ROC_AUC: {roc_auc:.4f}")
    print(f"PR_AUC: {pr_auc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"G-Mean: {gmean:.4f}")

# 计算平均结果和标准差
avg_roc_auc = np.mean(roc_auc_scores)
std_roc_auc = np.std(roc_auc_scores)
avg_pr_auc = np.mean(pr_auc_scores)
std_pr_auc = np.std(pr_auc_scores)
avg_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
avg_precision = np.mean(precision_scores)
std_precision = np.std(precision_scores)
avg_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)
avg_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
avg_mcc = np.mean(mcc_scores)
std_mcc = np.std(mcc_scores)
avg_gmean = np.mean(gmean_scores)
std_gmean = np.std(gmean_scores)

# 输出最终平均结果和标准差
print("\nFinal Average Results with Standard Deviation:")
print(f"Average ROC_AUC: {avg_roc_auc:.4f} ± {std_roc_auc:.4f}")
print(f"Average PR_AUC: {avg_pr_auc:.4f} ± {std_pr_auc:.4f}")
print(f"Average F1: {avg_f1:.4f} ± {std_f1:.4f}")
print(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}")
print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Average MCC: {avg_mcc:.4f} ± {std_mcc:.4f}")
print(f"Average G-Mean: {avg_gmean:.4f} ± {std_gmean:.4f}")
