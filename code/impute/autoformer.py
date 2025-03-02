# 'D:/L1000/6h_muldose_seq977.csv'
import pandas as pd
import numpy as np
from pygrinder import mcar
from pypots.imputation import Autoformer
from pypots.utils.metrics import calc_mae, calc_mse, calc_rmse, calc_mre
import os
import shutil
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr
if os.path.exists('autoformer'):
    shutil.rmtree('autoformer')

np.random.seed(42)

data = pd.read_csv('6h_muldose_seq977.csv', index_col=0)

data_array = data.to_numpy()
num_genes = data_array.shape[0]  # 基因数量
num_samples = data_array.shape[1]  # 样本数

num_doses = 4
num_genes_per_dose = num_genes

reshaped_data = np.zeros((num_samples // num_doses, num_doses, num_genes_per_dose))

for i in range(num_samples // num_doses):
    for j in range(num_doses):
        reshaped_data[i, j, :] = data_array[:, i * num_doses + j]


X = mcar(reshaped_data, 0.1)

dataset = {"X": X}

autoformer = Autoformer(
    n_steps=num_doses,  # 每个样本有四个剂量
    n_features=num_genes_per_dose,
    n_layers=2,
    d_model=256,  # d_model= n_heads * d_k
    d_ffn=256,  # 128
    n_heads=3,
    factor=2,
    moving_avg_window_size=2,
    dropout=0.1,
    epochs=1000,
    saving_path="autoformer",  # 设置保存路径
    model_saving_strategy="best"
)

# 训练模型
autoformer.fit(dataset)

# 填补缺失值
imputation = autoformer.impute(dataset)


indicating_mask = np.isnan(X) ^ np.isnan(reshaped_data)
mae = calc_mae(imputation, reshaped_data, indicating_mask)
mse = calc_mse(imputation, reshaped_data, indicating_mask)
rmse = calc_rmse(imputation, reshaped_data, indicating_mask)
mre = calc_mre(imputation, reshaped_data, indicating_mask)

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



saving_path = "data/autoformer"
os.makedirs(saving_path, exist_ok=True)
# 逆向重塑，将 (2630, 4, 977) 转换回 (977, 10520)
num_samples_restored = imputation.shape[0] * num_doses
restored_imputation = np.zeros((num_genes, num_samples_restored))
restored_X = np.zeros((num_genes, num_samples_restored))

for i in range(imputation.shape[0]):
    for j in range(num_doses):
        restored_imputation[:, i * num_doses + j] = imputation[i, j, :]
        restored_X[:, i * num_doses + j] = X[i, j, :]


imputation_df = pd.DataFrame(restored_imputation, index=data.index, columns=data.columns)
X_df = pd.DataFrame(restored_X, index=data.index, columns=data.columns)

imputation_df.to_csv(os.path.join(saving_path, 'imputation_results.csv'))
X_df.to_csv(os.path.join(saving_path, 'missing_values.csv'))
