import pandas as pd
import numpy as np
import random
np.set_printoptions(suppress=True)

phychem_file = "data0825/feature/drugfeature2_phychem_extract.csv"
finger_file = "data0825/feature/drugfeature1_finger_extract.csv"

# cell_line_files=["A375", "A549", "BT20", "HCT116", "HS578T", "HT29", "LNCAP", "LOVO", "MCF7", "MDAMB231", "PC3", "RKO", "SKMEL28", "SW620", "VCAP"]
cell_line_files = ["A375", "A549", "BT20", "HCT116", "HS578T", "HT29", "LOVO", "MCF7", "MDAMB231", "PC3", "RKO",
                   "SKBR3", "SKMEL28", "SW620", "SW948", "VCAP"]


cell_line_path_before = "data0825/feature/GP_977_final/"
cell_line_path_after = "data0825/feature/GP_3908_final/"
drugdrug_file = "data0825/feature/DD.csv"
cell_line_feature = "data0825/feature/16cell_zerodose_977cellname.csv"


def load_data(cell_line_name="all", score="S", dataid=2):
    '''
    cell_line_name: control which cell lines to load
    acceptable parameter types:list or str
        if "all"(default), load all cell lines
        if "cell line name"(e.g., "HT29"), load this cell line
        if list (e.g., ["HT29","A375"]), load "HT29" and "A375"
    '''
    extract = pd.read_csv(drugdrug_file, usecols=[3, 4, 5, 6, 7, 8, 9, 10, 11])
    phychem=pd.read_csv(phychem_file)
    finger=pd.read_csv(finger_file)
    cell_line = pd.read_csv(cell_line_feature)  # 未处理细胞系的基因表达谱（CGE）
    # column_name=list(finger.columns)
    # column_name[0]="drug_id"
    # finger.columns=column_name

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
    # data = np.zeros((n_sample, 8794))
    # data = np.zeros((n_sample, 4804))#2  （977+811+55）*2+977=4803
    data = np.zeros((n_sample, 10666))  # 3 （3908+811+55）*2+977=10665
    # data = np.zeros((n_sample, 1955))  # 4   977*2=1954
    # data = np.zeros((n_sample, 7817))  # 5   3908*2=7816
    dataid = dataid
    for i in range(n_sample):
        drugA_id = drug_comb.at[i, "drug_row_cid"]
        drugB_id = drug_comb.at[i, "drug_col_cid"]
        drugA_finger = get_finger(finger, drugA_id)
        drugB_finger = get_finger(finger, drugB_id)
        drugA_phychem = get_phychem(phychem, drugA_id)
        drugB_phychem = get_phychem(phychem, drugB_id)
        # drugA_tox = get_tox(tox, drugA_id)
        # drugB_tox = get_tox(tox, drugB_id)
        cell_line_name = drug_comb.at[i, "cell_line_name"]
        drugA_express_before = get_express(all_express_before[cell_line_name], drugA_id)
        drugB_express_before = get_express(all_express_before[cell_line_name], drugB_id)
        cell_line_name = drug_comb.at[i, "cell_line_name"]
        drugA_express_after = get_express(all_express_after[cell_line_name], drugA_id)
        drugB_express_after = get_express(all_express_after[cell_line_name], drugB_id)
        cell_line_name = drug_comb.at[i, "cell_line_name"]
        feature = get_cell_feature(cell_line, cell_line_name)
        label = drug_comb.at[i, "synergy_loewe"]


        if dataid == 2:  # 原特征（插补前）
            sample = np.hstack((drugA_express_before, drugA_finger, drugA_phychem,
                                drugB_express_before, drugB_finger, drugB_phychem,
                                feature))  # data3_1956
        if dataid == 3:  ##原特征（插补后）
            sample = np.hstack((drugA_express_after, drugA_finger, drugA_phychem,
                                drugB_express_after, drugB_finger, drugB_phychem,
                                feature))
        if dataid == 4:  # 插补前
            sample = np.hstack((drugA_express_before, drugB_express_before))  # data3_1956
        if dataid == 5:  # 插补后
            sample = np.hstack((drugA_express_after, drugB_express_after))  # data3_1956
#特征对齐
        if sample.shape[0] < 10665:
            random_values = [random.uniform(-500, 500) for _ in range(10665 - sample.shape[0])]
            sample = np.hstack(( random_values,sample))

        sample = np.hstack((sample, label))



        data[i]=sample
    return data


def get_finger(finger, drug_id):
    drug_finger = finger.loc[finger['cid'] == drug_id]
    drug_finger = np.array(drug_finger)
    drug_finger = drug_finger.reshape(drug_finger.shape[1])[1:]
    return drug_finger


def get_phychem(phychem, drug_id):
    drug_phychem = phychem.loc[phychem["cid"] == drug_id]
    drug_phychem = np.array(drug_phychem)
    drug_phychem = drug_phychem.reshape(drug_phychem.shape[1])[1:]
    return drug_phychem


# def get_express(express,drug_id):
#     drug_express=express[str(drug_id)]
#     drug_express=np.array(drug_express)
#     return drug_express
def get_express(express, drug_id):
    # 将 express 数据框的列名转换为整数
    express.columns = express.columns.astype(str).str.replace(r'\.0$', '', regex=True)
    if str(drug_id) not in express.columns.values:
        return None
    drug_express = express[str(drug_id)]
    drug_express = np.array(drug_express)
    # print(drug_express.shape)
    return drug_express


def get_cell_feature(feature, cell_line_name):
    cell_feature = feature[str(cell_line_name)]
    cell_feature = np.array(cell_feature)
    return cell_feature

