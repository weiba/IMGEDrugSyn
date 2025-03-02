
import pandas as pd
import numpy as np
import random
import itertools
seed = 42
np.random.seed(seed)
np.set_printoptions(suppress=True)
np.set_printoptions(suppress=True)

phychem_file = "data20210125/ChemoPy.csv"
finger_file = "data20210125/ECFP6.csv"
tox_file = "data20210125/1RDKit_tox.csv"

# cell_line_files=["A375", "A549", "BT20", "HCT116", "HS578T", "HT29", "LNCAP", "LOVO", "MCF7", "MDAMB231", "PC3", "RKO", "SKMEL28", "SW620", "VCAP"]
cell_line_files = ["A375", "A549", "BT20", "HCT116", "HS578T", "HT29", "LOVO", "MCF7", "MDAMB231", "PC3", "RKO",
                   "SKBR3", "SKMEL28", "SW620", "SW948", "VCAP"]


cell_line_path_before = "data20210125/GP_977_final/"
cell_line_path_after = "data20210125/GP_3908_final/"
drugdrug_file = "data20210125/DD.csv"
cell_line_feature = "data20210125/16cell_zerodose_977cellname.csv"


def load_data(cell_line_name="all", score="S", dataid=2):
    extract = pd.read_csv(drugdrug_file, usecols=[3, 4, 5, 11])  # dd (3192,4)
    phychem = pd.read_csv(phychem_file)  # 药物理化性质（DPP）(153,56)
    finger = pd.read_csv(finger_file)  # 药物分子指纹（DMF）(153,882)
    tox = pd.read_csv(tox_file)#药物的毒物特征（68，12）
    cell_line = pd.read_csv(cell_line_feature)  # 未处理细胞系的基因表达谱（CGE）
    # column_name=list(finger.columns)
    # column_name[0]="drug_id"
    # finger.columns=column_name
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
        drugA_finger = get_finger(finger, drugA_id)
        drugB_finger = get_finger(finger, drugB_id)
        drugA_phychem = get_phychem(phychem, drugA_id)
        drugB_phychem = get_phychem(phychem, drugB_id)
        drugA_tox = get_tox(tox, drugA_id)
        drugB_tox = get_tox(tox, drugB_id)
        cell_line_name = drug_comb.at[i, "cell_line_name"]
        drugA_express_before = get_express(all_express_before[cell_line_name], drugA_id)
        drugB_express_before = get_express(all_express_before[cell_line_name], drugB_id)
        cell_line_name = drug_comb.at[i, "cell_line_name"]
        drugA_express_after = get_express(all_express_after[cell_line_name], drugA_id)
        drugB_express_after = get_express(all_express_after[cell_line_name], drugB_id)
        cell_line_name = drug_comb.at[i, "cell_line_name"]
        feature = get_cell_feature(cell_line, cell_line_name)
        label = drug_comb.at[i, "label"]

        #维度统一
        XXX = np.hstack((drugA_express_after, drugA_finger, drugA_phychem, drugA_tox,
                            drugB_express_after, drugB_finger, drugB_phychem, drugB_tox,
                            feature))#18257
        target_feature_dim = XXX.shape[0]  # 记录目标特征维度



        if dataid == 0:  # DeepSynergy原始特征（无毒物）10435
            sample = np.hstack((drugA_finger, drugA_phychem, drugB_finger, drugB_phychem,
                                feature))
        if dataid == 1:  # DeepSynergy原始特征  10459   auc5496 aupr1752 f1 21 recall 1176
            sample = np.hstack((drugA_finger, drugA_phychem, drugA_tox, drugB_finger, drugB_phychem, drugB_tox,
                                feature))
        if dataid == 2:  # 在DeepSynergy原基础上增加了GP数据12413   auc7622 aupr4002 f1 20  recall 1176
            sample = np.hstack((drugA_express_before, drugA_finger, drugA_phychem, drugA_tox,
                                drugB_express_before, drugB_finger, drugB_phychem, drugB_tox,
                                feature))  # data3_1956
        if dataid == 3:  ##在DeepSynergy原基础上增加了GP插补后的数据
            sample = np.hstack((drugA_express_after, drugA_finger, drugA_phychem, drugA_tox,
                                drugB_express_after, drugB_finger, drugB_phychem, drugB_tox,
                                feature))
        if dataid == 4:  # 插补前
            sample = np.hstack((drugA_express_before, drugB_express_before))  # data3_1956
        if dataid == 5:  # 插补后
            sample = np.hstack((drugA_express_after, drugB_express_after))  # data3_1956

# 统一维度实验！
        if target_feature_dim is not None:
            if sample.shape[0] < target_feature_dim:
                random_values = [random.uniform(-500, 500) for _ in range(target_feature_dim - sample.shape[0])]
                # random_values = np.random.rand(target_feature_dim - sample.shape[0])
                sample = np.hstack(( random_values,sample))
                # sample = np.hstack((sample,random_values ))

        sample = np.hstack((sample, label))  # data3_1956

        data.append(sample)
        print(f"Sample {i}: {sample.shape}")
    print("***************load data-{}***************".format(dataid))
    data = np.array(data)
    return data[:, 0:-1], data[:, -1], dataid


def get_finger(finger, drug_id):
    drug_finger = finger.loc[finger['cid'] == drug_id]
    drug_finger = np.array(drug_finger)
    drug_finger = drug_finger.reshape(drug_finger.shape[1])[1:]
    # print(drug_finger.shape)
    return drug_finger


def get_phychem(phychem, drug_id):
    drug_phychem = phychem.loc[phychem["cid"] == drug_id]
    # print(drug_phychem)
    drug_phychem = np.array(drug_phychem)
    drug_phychem = drug_phychem.reshape(drug_phychem.shape[1])[1:]
    # print(drug_phychem.shape)
    return drug_phychem

def get_cell_feature(feature,cell_line_name):
    # print(feature.head())
    # print(cell_line_name)
    cell_feature=feature[str(cell_line_name)]
    cell_feature=np.array(cell_feature)
    return cell_feature
def get_express(express, drug_id):
    # 将 express 数据框的列名转换为整数
    express.columns = express.columns.astype(str).str.replace(r'\.0$', '', regex=True)
    if str(drug_id) not in express.columns.values:
        return None
    drug_express = express[str(drug_id)]
    drug_express = np.array(drug_express)
    # print(drug_express.shape)
    return drug_express
def get_tox(tox, drug_id):
    drug_tox = tox.loc[tox['cid'] == drug_id]
    drug_tox = np.array(drug_tox)
    drug_tox = drug_tox.reshape(drug_tox.shape[1])[1:]
    return drug_tox
def load():
    x,y,dataid=load_data()
    for index in range(len(y)):
        if y[index]==3.0:
            y[index]=0.0
        if y[index]==1.0:
            y[index]=0.0
        if y[index]==2.0:
            y[index]=1.0
    print("x_shape:",x.shape)
    print("y_shape:",y.shape)

    from sklearn.model_selection import train_test_split
    # 首先将数据集 x 和 y 按照 8:2 的比例划分成训练集和临时集 (train_temp_x, train_temp_y)
    train_x, temp_x, train_y, temp_y = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)

    # 将临时集按照 1:1 的比例划分成验证集和测试集 (val_x, val_y 和 test_x, test_y)
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=seed, stratify=temp_y)

    # 将训练集和验证集组合成一个新的集合 mm_x, mm_y
    mm_x = np.vstack((train_x, val_x))
    mm_y = np.hstack((train_y, val_y))

    # 输出各个集合的大小
    print("Train set x_shape:", train_x.shape)
    print("Train set y_shape:", train_y.shape)
    print("Validation set x_shape:", val_x.shape)
    print("Validation set y_shape:", val_y.shape)
    print("Test set x_shape:", test_x.shape)
    print("Test set y_shape:", test_y.shape)
    print("MM set x_shape:", mm_x.shape)
    print("MM set y_shape:", mm_y.shape)

    return train_x,val_x,mm_x,test_x,train_y,val_y,mm_y,test_y