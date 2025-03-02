import numpy as np
import pickle
import multiprocessing as mp

from gcForest import gcForest
from evaluation import R2, MSE
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

# 固定随机种子
def set_random_seed(seed=1):
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

set_random_seed(6)

def mp_kfold(x, y, n_spilt=5, n_repeat=1, random_state=4, n_jobs=10):  # n_repeat=3
    # metrics
    set_random_seed(random_state)
    mses, rmses, maes, r2s, pearsonrs, y_vals, y_preds = {}, {}, {}, {}, {}, {}, {}
    # train & test spilt, using k-folds cross-validation
    kf = RepeatedKFold(n_splits=n_spilt, n_repeats=n_repeat, random_state=random_state)
    cv = [(t, v) for (t, v) in kf.split(x)]
    # assign tasks for multiple process
    assignments = assign_tasks(n_spilt * n_repeat, n_jobs)
    pool = mp.Pool(n_jobs)
    tasks = [pool.apply_async(kfold_async, args=(x, y, k, task, cv)) for k, task in enumerate(assignments)]
    pool.close()
    pool.join()
    results = [task.get() for task in tasks]
    for result in results:
        mses.update(result[0])
        rmses.update(result[1])
        maes.update(result[2])
        r2s.update(result[3])
        pearsonrs.update(result[4])
        y_vals.update(result[5])
        y_preds.update(result[6])

    # 输出每一折的结果
    for i in range(len(mses)):
        print(f"Fold {i}:")
        print(f"MSE={mses[i]:.4f}, RMSE={rmses[i]:.4f}, MAE={maes[i]:.4f}, R^2={r2s[i]:.4f}, Pearson={pearsonrs[i]:.4f}")
        print("====================================")

    # 计算并输出五折的平均结果
    mses = sorted(mses.items(), key=lambda x: x[0])
    mses = [value[1] for value in mses]
    rmses = sorted(rmses.items(), key=lambda x: x[0])
    rmses = [value[1] for value in rmses]
    maes = sorted(maes.items(), key=lambda x: x[0])
    maes = [value[1] for value in maes]
    r2s = sorted(r2s.items(), key=lambda x: x[0])
    r2s = [value[1] for value in r2s]
    pearsonrs = sorted(pearsonrs.items(), key=lambda x: x[0])
    pearsonrs = [value[1] for value in pearsonrs]
    y_vals = sorted(y_vals.items(), key=lambda x: x[0])
    y_vals = [value[1] for value in y_vals]
    y_preds = sorted(y_preds.items(), key=lambda x: x[0])
    y_preds = [value[1] for value in y_preds]

    print("MSE mean={:.4f}, std={:.4f}".format(np.mean(mses), np.std(mses)))
    print("RMSE mean={:.4f}, std={:.4f}".format(np.mean(rmses), np.std(rmses)))
    print("MAE mean={:.4f}, std={:.4f}".format(np.mean(maes), np.std(maes)))
    print("R^2 mean={:.4f}, std={:.4f}".format(np.mean(r2s), np.std(r2s)))
    print("PEARSONR mean={:.4f}, std={:.4f}".format(np.mean(pearsonrs), np.std(pearsonrs)))

    return mses, rmses, maes, r2s, pearsonrs, y_vals, y_preds

def kfold_async(x, y, k, task, cv):
    begin, end = task[0], task[1]
    print("Process_{}: {}-{} folds".format(k, begin, end-1))
    mses, rmses, maes, r2s, pearsonrs, y_tests, y_preds = {}, {}, {}, {}, {}, {}, {}
    for i in range(begin, end):
        (train_id, test_id) = cv[i]
        x_train, x_test, y_train, y_test = x[train_id], x[test_id], y[train_id], y[test_id]
        gc = None
        config = get_config()
        gc = gcForest(config, i)
        gc.fit(x_train, y_train)
        y_pred = gc.predict(x_test)
        mses[i] = mean_squared_error(y_test, y_pred)
        rmses[i] = np.sqrt(mses[i])
        maes[i] = mean_absolute_error(y_test, y_pred)
        r2s[i] = r2_score(y_test, y_pred)
        pearsonrs[i] = pearsonr(y_test, y_pred)[0]
        y_tests[i] = y_test
        y_preds[i] = y_pred
        print(f"======={i}=======")
    return (mses, rmses, maes, r2s, pearsonrs, y_tests, y_preds)

def assign_tasks(n_fold, n_jobs):
    task_len = n_fold // n_jobs
    task_left = n_fold % n_jobs
    task_lens = [task_len for i in range(n_jobs)]
    for i in range(task_left):
        task_lens[i] += 1
    assignments = []
    begin = 0
    for i in range(len(task_lens)):
        assignments.append((begin, begin + task_lens[i]))
        begin = begin + task_lens[i]
    return assignments

def get_config():
    config = {}
    config["error_threshold"] = 0.2  # 0.3
    config["estimator_configs"] = []
    config["resampling_rate"] = 2
    for i in range(6):
        config["estimator_configs"].append({"n_fold": 5, "type": "GradientBoostingRegressor", "n_estimators": 50, "random_state": i})
    config["train_evaluation"] = R2
    config["early_stop_rounds"] = 1
    config["random_state"] = 4
    config["max_layers"] = 6  # 3
    return config

from data0825.load_data_mse import load_data
data = load_data()
print(data.shape)
x = data[:, 0:-1]
y = data[:, -1]
mses, rmses, maes, r2s, pearsonrs, y_tests, y_preds = mp_kfold(x, y, n_repeat=1, n_jobs=5)
