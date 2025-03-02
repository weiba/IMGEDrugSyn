import random
import pandas as pd
import numpy as np
import lightgbm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score

phychem_file = "data20210125/drugfeature2_phychem_extract.csv"
finger_file = "data20210125/drugfeature1_finger_extract.csv"

# cell_line_files=["A375", "A549", "BT20", "HCT116", "HS578T", "HT29", "LNCAP", "LOVO", "MCF7", "MDAMB231", "PC3", "RKO", "SKMEL28", "SW620", "VCAP"]
cell_line_files = ["A375", "A549", "BT20", "HCT116", "HS578T", "HT29", "LOVO", "MCF7", "MDAMB231", "PC3", "RKO",
                   "SKBR3", "SKMEL28", "SW620", "SW948", "VCAP"]

cell_line_feature = "data20210125/16cell_zerodose_977cellname.csv"
cell_line_path_before = "data20210125/GP_977_final/"
cell_line_path_after = "data20210125/GP_3908_final/"
drugdrug_file = "data20210125/DD.csv"
maccs = "data20210125/MACCS_fingerprints.csv"
morgan = "data20210125/Morgan_fingerprints.csv"
RDK = "data20210125/RDK_fingerprints.csv"

seed = 42
np.random.seed(seed)


def load_data(cell_line_name="all", score="S", dataid=5):
    extract = pd.read_csv(drugdrug_file, usecols=[3, 4, 5, 11])  # dd (3192,4)
    phychem1 = pd.read_csv(maccs)
    phychem2 = pd.read_csv(morgan)
    phychem3 = pd.read_csv(RDK)
    cell_line = pd.read_csv(cell_line_feature)  # 未处理细胞系的基因表达谱（CGE）

    label = pd.Categorical(extract["label"])
    extract["label"] = label.codes + 1
    # print(list(cell_line["A375"]))

    if cell_line_name == "all":  # 细胞系特异性药物诱导基因表达谱（DGE）
        all_express_before = {cell_line: pd.read_csv("{}{}.csv".format(cell_line_path_before, cell_line)) for cell_line in
                              cell_line_files}
    elif type(cell_line_name) is list:
        all_express_before = {cell_line: pd.read_csv("{}{}.csv".format(cell_line_path_before, cell_line)) for cell_line in
                              cell_line_name}
    elif type(cell_line_name) is str:
        all_express_before = {cell_line_name: pd.read_csv("{}{}.csv".format(cell_line_path_before, cell_line_name))}
    else:
        raise ValueError("Invalid parameter: {}".format(cell_line_name))

    if cell_line_name == "all":  # 细胞系特异性药物诱导基因表达谱（DGE）
        all_express_after = {cell_line: pd.read_csv("{}{}.csv".format(cell_line_path_after, cell_line)) for cell_line in
                             cell_line_files}
    elif type(cell_line_name) is list:
        all_express_after = {cell_line: pd.read_csv("{}{}.csv".format(cell_line_path_after, cell_line)) for cell_line in
                             cell_line_name}
    elif type(cell_line_name) is str:
        all_express_after = {cell_line_name: pd.read_csv("{}{}.csv".format(cell_line_path_after, cell_line_name))}
    else:
        raise ValueError("Invalid parameter: {}".format(cell_line_name))

    drug_comb = None
    if cell_line_name == "all":
        drug_comb = extract
    else:
        if type(cell_line_name) is list:
            drug_comb = extract.loc[extract["cell_line_name"].isin(cell_line_name)]
        else:
            drug_comb = extract.loc[extract["cell_line_name"] == cell_line_name]

    n_sample = drug_comb.shape[0]
    # n_feature=((phychem.shape[1]-1)+(finger.shape[1]-1)+978)*2+1+978
    drug_comb.index = range(n_sample)
    # data=np.zeros((n_sample,n_feature))
    data = []
    dataid = dataid
    for i in range(n_sample):
        drugA_id = drug_comb.at[i, "drug_row_cid"]
        drugB_id = drug_comb.at[i, "drug_col_cid"]
        # drugA_finger=get_finger(finger,drugA_id)
        # drugB_finger=get_finger(finger,drugB_id)
        drugA_phychem1 = get_phychem(phychem1, drugA_id)
        drugB_phychem1 = get_phychem(phychem1, drugB_id)
        drugA_phychem2 = get_phychem(phychem2, drugA_id)
        drugB_phychem2 = get_phychem(phychem2, drugB_id)
        drugA_phychem3 = get_phychem(phychem3, drugA_id)
        drugB_phychem3 = get_phychem(phychem3, drugB_id)

        cell_line_name = drug_comb.at[i, "cell_line_name"]
        drugA_express_before = get_express(all_express_before[cell_line_name], drugA_id)
        drugB_express_before = get_express(all_express_before[cell_line_name], drugB_id)
        cell_line_name = drug_comb.at[i, "cell_line_name"]
        drugA_express_after = get_express(all_express_after[cell_line_name], drugA_id)
        drugB_express_after = get_express(all_express_after[cell_line_name], drugB_id)
        cell_line_name = drug_comb.at[i, "cell_line_name"]
        feature = get_cell_feature(cell_line, cell_line_name)

        label = drug_comb.at[i, "label"]

        # if dataid==1:
        #     sample=np.hstack((drugA_finger,drugB_finger,feature,label))                             #data1_2740
        # elif dataid==2:
        #     sample=np.hstack((drugA_phychem,drugB_phychem,feature,label))                           #data2_1088

        if dataid == 1:  # 原特征
            sample = np.hstack((drugA_phychem1, drugA_phychem2, drugA_phychem3,
                                drugB_phychem1, drugB_phychem2, drugB_phychem3,
                                feature))  # 7457
        if dataid == 2:  # 原特征+插补前  9409
            sample = np.hstack((drugA_phychem1, drugA_phychem2, drugA_phychem3, drugA_express_before,
                                drugB_phychem1, drugB_phychem2, drugB_phychem3, drugB_express_before,
                                feature))  #
        if dataid == 3:  # 原特征+插补后  15271
            sample = np.hstack((drugA_phychem1, drugA_phychem2, drugA_phychem3, drugA_express_after,
                                drugB_phychem1, drugB_phychem2, drugB_phychem3, drugB_express_after,
                                feature))  # data3_1956
        if dataid == 4:  # 插补前
            sample = np.hstack((drugA_express_before, drugB_express_before))  # data3_1956
        if dataid == 5:  # 插补后
            sample = np.hstack((drugA_express_after, drugB_express_after))  # data3_1956
        #
        # #统一维度
        #         if sample.shape[0] < 15271:
        #             random_values = [random.uniform(-500, 500) for _ in range(15271 - sample.shape[0])]
        #             sample = np.hstack(( random_values,sample))

        sample = np.hstack((sample, label))  # data3_1956

        data.append(sample)
        print(f"Sample {i}: {sample.shape}")
    print("***************load data-{}***************".format(dataid))
    data = np.array(data)
    return data[:, 0:-1], data[:, -1], dataid


def get_finger(finger, drug_id):
    drug_finger = finger.loc[finger['drug_id'] == drug_id]
    drug_finger = np.array(drug_finger)
    drug_finger = drug_finger.reshape(drug_finger.shape[1])[1:]
    # print(drug_finger.shape)
    return drug_finger


def get_cell_feature(feature, cell_line_name):
    # print(feature.head())
    # print(cell_line_name)
    cell_feature = feature[str(cell_line_name)]
    cell_feature = np.array(cell_feature)
    return cell_feature


def get_phychem(phychem, drug_id):
    drug_phychem = phychem.loc[phychem["cid"] == drug_id]
    # print(drug_phychem)
    drug_phychem = np.array(drug_phychem)
    drug_phychem = drug_phychem.reshape(drug_phychem.shape[1])[1:]
    # print(drug_phychem.shape)
    return drug_phychem


def get_express(express, drug_id):
    # 将 express 数据框的列名转换为整数
    express.columns = express.columns.astype(str).str.replace(r'\.0$', '', regex=True)
    if str(drug_id) not in express.columns.values:
        return None
    drug_express = express[str(drug_id)]
    drug_express = np.array(drug_express)
    # print(drug_express.shape)
    return drug_express


# 用于存储所有五次五折交叉验证的结果
all_fold_results = []

for repeat in range(5):
    x, y, dataid = load_data()
    for index in range(len(y)):
        if y[index] == 3.0:
            y[index] = 0.0
        if y[index] == 1.0:
            y[index] = 0.0
        if y[index] == 2.0:
            y[index] = 1.0
    print("x_shape:", x.shape)
    print("y_shape:", y.shape)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    # 用于存储每一折的评估结果
    fold_results = []

    # 五折交叉验证
    for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
        # 划分训练集和测试集
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_data = lightgbm.Dataset(x_train, label=y_train)

        # 设置LightGBM的参数
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'num_leaves': 20,
            'max_depth': 8,
            'force_col_wise': 'true',
            'learning_rate': 0.05,
            'verbose': 0,
            'n_estimators': 1000,
            'reg_alpha': 2,
            'seed': seed
        }

        # 训练模型
        model = lightgbm.train(params, train_data, num_boost_round=100)

        # 在测试集上进行预测
        y_pred_prob = model.predict(x_test)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # 计算各项评估指标
        auc = roc_auc_score(y_test, y_pred_prob)
        aupr = average_precision_score(y_test, y_pred_prob)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # 存储本折的结果
        fold_results.append({
            "Fold": fold,
            "AUC": auc,
            "AUPR": aupr,
            "Accuracy": acc,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall
        })

    all_fold_results.extend(fold_results)


results_df = pd.DataFrame(all_fold_results)

# 计算每个指标的平均值和方差
summary = results_df.iloc[:, 1:].agg(['mean', 'std'])

average_per_fold = results_df.groupby('Fold').mean()
# 输出每折的平均结果
print("\nAverage Results per Fold over 5 repeats of 5 folds:")
print(average_per_fold)
# 输出五次五折的平均值和方差
print("\nOverall Results (Mean and Std over 5 repeats of 5 folds):")
print(summary)
