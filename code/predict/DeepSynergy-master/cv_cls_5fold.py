import os
import sys
import pandas as pd
import numpy as np
import pickle
import gzip
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, \
    average_precision_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import data_load_org_same

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # specify GPU
import keras as K
import tensorflow as tf
from keras import backend
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout

hyperparameter_file = 'hyperparameters'  # textfile which contains the hyperparameters of the model

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

exec(open(hyperparameter_file).read())
X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = data_load_org_same.load()

# Setup TensorFlow session
config = tf.ConfigProto(
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(allow_growth=True))
set_session(tf.Session(config=config))

# 5-fold Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store results for all folds
all_fold_accuracies, all_fold_precisions, all_fold_recalls = [], [], []
all_fold_f1_scores, all_fold_aucs, all_fold_auprs = [], [], []

# Perform 5-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}/5")

    # Split data into training and validation sets for the current fold
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # Build and compile the model
    model = Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(Dense(layers[i], input_shape=(X_fold_train.shape[1],), activation=act_func,
                            kernel_initializer='he_normal'))
            model.add(Dropout(float(input_dropout)))
        elif i == len(layers) - 1:
            model.add(Dense(layers[i], activation='sigmoid',
                            kernel_initializer="he_normal"))  # sigmoid activation for binary classification
        else:
            model.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
            model.add(Dropout(float(dropout)))

    model.compile(loss='binary_crossentropy',  # binary crossentropy for binary classification
                  optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))

    # Train the model
    hist = model.fit(X_fold_train, y_fold_train, epochs=epochs, shuffle=True, batch_size=64,
                     validation_data=(X_fold_val, y_fold_val))
    val_loss = hist.history['val_loss']

    # Smooth validation loss
    model.reset_states()
    average_over = 15
    mov_av = moving_average(np.array(val_loss), average_over)
    smooth_val_loss = np.pad(mov_av, int(average_over / 2), mode='edge')
    epo = np.argmin(smooth_val_loss)

    # Retrain the model on the full training set
    hist = model.fit(X_fold_train, y_fold_train, epochs=epo, shuffle=True, batch_size=64,
                     validation_data=(X_fold_val, y_fold_val))
    # test_loss = hist.history['val_loss']

    # Test the model
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    aupr = average_precision_score(y_test, y_pred_prob)

    # Store fold results
    all_fold_accuracies.append(accuracy)
    all_fold_aucs.append(auc)
    all_fold_auprs.append(aupr)

    print(
        f"Fold {fold + 1} results: Accuracy: {accuracy:.4f},  AUC: {auc:.4f}, AUPR: {aupr:.4f}")

# After 5 folds, calculate average and standard deviation of metrics
avg_accuracy = np.mean(all_fold_accuracies)
std_accuracy = np.std(all_fold_accuracies)

avg_auc = np.mean(all_fold_aucs)
std_auc = np.std(all_fold_aucs)

avg_aupr = np.mean(all_fold_auprs)
std_aupr = np.std(all_fold_auprs)

# Output the results of each fold
print("\nResults from each fold:")
for i in range(5):
    print(f"Fold {i+1}: Accuracy: {all_fold_accuracies[i]:.4f}, Precision: {all_fold_precisions[i]:.4f}, "
          f"Recall: {all_fold_recalls[i]:.4f}, F1: {all_fold_f1_scores[i]:.4f}, AUC: {all_fold_aucs[i]:.4f}, "
          f"AUPR: {all_fold_auprs[i]:.4f}")

# Output the average and standard deviation of metrics
print("\nAverage and Standard Deviation across 5 folds:")
print(f"Average Accuracy: {avg_accuracy:.4f}, Std Accuracy: {std_accuracy:.4f}")
print(f"Average AUC: {avg_auc:.4f}, Std AUC: {std_auc:.4f}")
print(f"Average AUPR: {avg_aupr:.4f}, Std AUPR: {std_aupr:.4f}")

# Optionally, save results to a CSV file
results_df = pd.DataFrame({
    "Fold": list(range(1, 6)),
    "Accuracy": all_fold_accuracies,
    "AUC": all_fold_aucs,
    "AUPR": all_fold_auprs
})

results_df.to_csv("deep_synergy_cross_validation_results.csv", index=False)

# Save the final model (optional)
model.save("deep_synergy_model_final.h5")
print("模型已保存")
