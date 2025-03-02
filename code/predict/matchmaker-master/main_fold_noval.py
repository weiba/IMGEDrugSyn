import os
import numpy as np
import pandas as pd
import tensorflow as tf
import MatchMaker_noval
import performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 参数
num_folds = 5
base_path = "data/"  # 根据实际文件路径修改
results = {"pred1": [], "pred2": [], "pred_mean": []}  # 存储每种方式的结果
fold_results = []  # 存储每一折的结果

# 配置 MatchMaker 模型参数
num_cores = 8
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=num_cores,
    inter_op_parallelism_threads=num_cores,
    allow_soft_placement=True,
    device_count={'CPU': 1, 'GPU': 1}
)
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# 固定参数
norm = 'tanh_norm'
architecture = pd.read_csv('architecture.txt')
layers = {
    'DSN_1': architecture['DSN_1'][0],
    'DSN_2': architecture['DSN_2'][0],
    'SPN': architecture['SPN'][0]
}
l_rate = 0.0001
inDrop = 0.1
drop = 0.2
max_epoch = 30
batch_size = 16
model_name = "matchmaker.h5"
#-------------------------------------------------------------------------------------
#改变特征输入
#data1_977.csv
#data1.csv
#data1_633_ChemoPy.csv
#data1_1610_ChemoPy+GPno.csv
#data1_4541_ChemoPy+GP.csv

#data2_977.csv
#data2.csv
#data2_633_ChemoPy.csv
#data2_1610_ChemoPy+GPno.csv
#data2_4541_ChemoPy+GP.csv

# 加载数据data_loader
chem1, chem2, cell_line, synergies = MatchMaker_noval.data_loader(
    "data/data1_4541_ChemoPy+GP.csv",
    "data/data2_4541_ChemoPy+GP.csv",
    "data/cell_line_gex_16.csv", "data/Mydd.tsv"
)

# 五折交叉验证
for fold in range(1, num_folds + 1):
    print(f"\n第 {fold} 折结果:")

    # 加载训练和测试索引
    train_ind = os.path.join(base_path, f"train_inds_fold_{fold}.txt")
    test_ind = os.path.join(base_path, f"test_inds_fold_{fold}.txt")

    train_data, test_data = MatchMaker_noval.prepare_data_no_val(
        chem1, chem2, cell_line, synergies, norm,
        train_ind, test_ind
    )

    # 计算加权 MSE 损失的权重
    min_s = np.amin(train_data['y'])
    loss_weight = np.log(train_data['y'] - min_s + np.e)

    # 生成模型
    model = MatchMaker_noval.generate_network(train_data, layers, inDrop, drop)

    # 训练模型
    model = MatchMaker_noval.trainer_no_val(
        model, l_rate, train_data, max_epoch, batch_size,
        model_name, loss_weight
    )

    # 加载最佳模型
    model.load_weights(model_name)

    # 预测结果
    pred1 = MatchMaker_noval.predict(model, [test_data['drug1'], test_data['drug2']])
    pred2 = MatchMaker_noval.predict(model, [test_data['drug2'], test_data['drug1']])
    pred_mean = (pred1 + pred2) / 2

    # 计算评价指标
    mae1 = mean_absolute_error(test_data['y'], pred1)
    mse1 = mean_squared_error(test_data['y'], pred1)
    spearman1 = performance_metrics.spearman(test_data['y'], pred1)
    pearson1 = performance_metrics.pearson(test_data['y'], pred1)

    mae2 = mean_absolute_error(test_data['y'], pred2)
    mse2 = mean_squared_error(test_data['y'], pred2)
    spearman2 = performance_metrics.spearman(test_data['y'], pred2)
    pearson2 = performance_metrics.pearson(test_data['y'], pred2)

    mae_mean = mean_absolute_error(test_data['y'], pred_mean)
    mse_mean = mean_squared_error(test_data['y'], pred_mean)
    spearman_mean = performance_metrics.spearman(test_data['y'], pred_mean)
    pearson_mean = performance_metrics.pearson(test_data['y'], pred_mean)

    # 打印每折结果
    print(f"Pred1 - MAE: {mae1:.4f}, MSE: {mse1:.4f}, Spearman: {spearman1:.4f}, Pearson: {pearson1:.4f}")
    print(f"Pred2 - MAE: {mae2:.4f}, MSE: {mse2:.4f}, Spearman: {spearman2:.4f}, Pearson: {pearson2:.4f}")
    print(
        f"Pred Mean - MAE: {mae_mean:.4f}, MSE: {mse_mean:.4f}, Spearman: {spearman_mean:.4f}, Pearson: {pearson_mean:.4f}")

    # 存储每折结果
    fold_results.append({
        'pred1': [mae1, mse1, spearman1, pearson1],
        'pred2': [mae2, mse2, spearman2, pearson2],
        'pred_mean': [mae_mean, mse_mean, spearman_mean, pearson_mean]
    })

    # 存储结果
    results["pred1"].append([mae1, mse1, spearman1, pearson1])
    results["pred2"].append([mae2, mse2, spearman2, pearson2])
    results["pred_mean"].append([mae_mean, mse_mean, spearman_mean, pearson_mean])

# 输出五折每一折的结果
print("\n所有折的详细结果:")
for fold_idx, fold_result in enumerate(fold_results, 1):
    print(f"\n第 {fold_idx} 折:")
    for key, values in fold_result.items():
        print(
            f"{key} - MAE: {values[0]:.4f}, MSE: {values[1]:.4f}, Spearman: {values[2]:.4f}, Pearson: {values[3]:.4f}")

# 计算均值和方差
for key, values in results.items():
    metrics = np.array(values)
    mean_values = np.mean(metrics, axis=0)
    std_values = np.std(metrics, axis=0)
    print(f"\n{key} 的总均值和方差:")
    print(f"MAE: {mean_values[0]:.4f} ± {std_values[0]:.4f}")
    print(f"MSE: {mean_values[1]:.4f} ± {std_values[1]:.4f}")
    print(f"Spearman: {mean_values[2]:.4f} ± {std_values[2]:.4f}")
    print(f"Pearson: {mean_values[3]:.4f} ± {std_values[3]:.4f}")
