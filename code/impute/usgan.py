# 'D:/L1000/6h_muldose_seq977.csv'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
import pygrinder
from pypots.imputation import USGAN
from pypots.imputation import iTransformer
from pypots.utils.metrics import calc_mae, calc_mse, calc_rmse, calc_mre
import os
import shutil
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr
if os.path.exists('usgan'):
    shutil.rmtree('usgan')

# 设置随机种子
np.random.seed(42)
data = pd.read_csv('6h_muldose_seq977.csv', index_col=0)  # 假设第一列为索引列，即基因名称

# 将数据转换为 numpy 数组
data_array = data.to_numpy()

# 数据集信息
num_genes = data_array.shape[0]  # 基因数量
num_samples = data_array.shape[1]  # 样本数（每个剂量下的数据）

num_doses = 4
num_genes_per_dose = num_genes

# 初始化一个新的数组来存储重塑后的数据
reshaped_data = np.zeros((num_samples // num_doses, num_doses, num_genes_per_dose))

# 将每四列数据按顺序填充到 reshaped_data 中
for i in range(num_samples // num_doses):
    for j in range(num_doses):
        reshaped_data[i, j, :] = data_array[:, i * num_doses + j]

# 创建一个虚拟的缺失值数据，这里假设随机隐藏10%的观测值
X = mcar(reshaped_data, 0.1)

#block_missing用法：(X,factor,block_len,block_len,feature_idx=None,feature_idx=None)
    #X: 输入的数据
    #factor: 控制缺失值的因子，用于计算缺失的比率。
    #block_len: 每个缺失块的长度（时间步数）
    #block_width: 每个缺失块的宽度（特征数）
        #注意：后两个参数，是指的开始位置。eg：如果写[0,1]那么就是从索引0或者1开始清零block_len行。而不是清零0和1行。而是清零01或者12行
    #feature_idx: 可选参数，指定应用缺失块的特征索引。
    #step_idx: 可选参数，指定应用缺失块的时间步索引。
# X = pygrinder.block_missing(reshaped_data,0.1,3,977,step_idx=[0])
# 准备模型输入的字典格式
dataset = {"X": X}



usgan = USGAN(
    n_steps=num_doses,  # 每个样本有四个剂量
    n_features=num_genes_per_dose,
    rnn_hidden_size=256,
    dropout=0.1,
    epochs=1000,
    saving_path="usgan",  # 设置保存路径
    model_saving_strategy="best"
)




# 训练模型
usgan.fit(dataset)

# 填补缺失值
imputation = usgan.impute(dataset)


indicating_mask = np.isnan(X) ^ np.isnan(reshaped_data)  # 生成指示掩码用于计算填补误差
mae = calc_mae(imputation, reshaped_data, indicating_mask)
mse = calc_mse(imputation, reshaped_data, indicating_mask)
rmse = calc_rmse(imputation, reshaped_data, indicating_mask)
mre = calc_mre(imputation, reshaped_data, indicating_mask)
# 计算皮尔逊相关系数和R2
imputed_values = imputation[indicating_mask]
true_values = reshaped_data[indicating_mask]

pearson_corr, _ = pearsonr(imputed_values, true_values)
r2 = r2_score(true_values, imputed_values)



print(f"Mean Absolute Error (MAE) on imputed data: {mae}")
print(f"Mean Squared Error (MSE) on imputed data: {mse}")
print(f"Root Mean Squared Error (RMSE) on imputed data: {rmse}")
print(f"Mean Relative Error (MRE) on imputed data: {mre}")
print(f"Pearson Correlation Coefficient on imputed data: {pearson_corr}")
print(f"R2 Score on imputed data: {r2}")


# 设置保存路径
saving_path = "data/usgan"
os.makedirs(saving_path, exist_ok=True)  # 如果路径不存在，则创建它
# 逆向重塑，将 (2630, 4, 977) 转换回 (977, 10520)
num_samples_restored = imputation.shape[0] * num_doses
restored_imputation = np.zeros((num_genes, num_samples_restored))
restored_X = np.zeros((num_genes, num_samples_restored))

for i in range(imputation.shape[0]):
    for j in range(num_doses):
        restored_imputation[:, i * num_doses + j] = imputation[i, j, :]
        restored_X[:, i * num_doses + j] = X[i, j, :]

# 保存插补后的结果和原始数据 X 到 CSV 文件
imputation_df = pd.DataFrame(restored_imputation, index=data.index, columns=data.columns)
X_df = pd.DataFrame(restored_X, index=data.index, columns=data.columns)

imputation_df.to_csv(os.path.join(saving_path, 'imputation_results.csv'))
X_df.to_csv(os.path.join(saving_path, 'missing_values.csv'))
