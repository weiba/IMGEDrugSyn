import pandas as pd
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from helper_funcs import normalize

def data_loader(drug1_chemicals,drug2_chemicals,cell_line_gex,comb_data_name):
    print("File reading ...")
    comb_data = pd.read_csv(comb_data_name, sep="\t")
    cell_line = pd.read_csv(cell_line_gex,header=None)
    chem1 = pd.read_csv(drug1_chemicals,header=None)
    chem2 = pd.read_csv(drug2_chemicals,header=None)
    synergies = np.array(comb_data["synergy_loewe"])#回归

    cell_line = np.array(cell_line.values)
    chem1 = np.array(chem1.values)
    chem2 = np.array(chem2.values)

# # 目标特征维度
#     target_feature_dim = 4541
#
#     if chem1.shape[1] < target_feature_dim:
#         random_values = np.random.uniform(-500, 500, size=(chem1.shape[0], target_feature_dim - chem1.shape[1]))
#         chem1 = np.hstack((chem1, random_values))
#
#     if chem2.shape[1] < target_feature_dim:
#         random_values = np.random.uniform(-500, 500, size=(chem2.shape[0], target_feature_dim - chem2.shape[1]))
#         chem2 = np.hstack((chem2, random_values))

    return chem1, chem2, cell_line, synergies


def prepare_data_no_val(chem1, chem2, cell_line, synergies, norm, train_ind_fname, test_ind_fname):
    print("Data normalization and preparation of train/test data")
    test_ind = list(np.loadtxt(test_ind_fname, dtype=int))
    train_ind = list(np.loadtxt(train_ind_fname, dtype=int))

    train_data = {}
    test_data = {}
    train1 = np.concatenate((chem1[train_ind, :], chem2[train_ind, :]), axis=0)
    train_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(train1, norm=norm)
    test_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(chem1[test_ind, :], mean1, std1, mean2, std2,
                                                                       feat_filt=feat_filt, norm=norm)
    train2 = np.concatenate((chem2[train_ind, :], chem1[train_ind, :]), axis=0)
    train_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(train2, norm=norm)
    test_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(chem2[test_ind, :], mean1, std1, mean2, std2,
                                                                       feat_filt=feat_filt, norm=norm)

    # ------------细胞系特征--------------要取消的时候 把这一段注释即可
    train3 = np.concatenate((cell_line[train_ind,:],cell_line[train_ind,:]),axis=0)
    train_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(train3, norm=norm)
    test_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(cell_line[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    train_data['drug1'] = np.concatenate((train_data['drug1'],train_cell_line),axis=1)
    train_data['drug2'] = np.concatenate((train_data['drug2'],train_cell_line),axis=1)
    test_data['drug1'] = np.concatenate((test_data['drug1'],test_cell_line),axis=1)
    test_data['drug2'] = np.concatenate((test_data['drug2'],test_cell_line),axis=1)


    train_data['y'] = np.concatenate((synergies[train_ind], synergies[train_ind]), axis=0)
    test_data['y'] = synergies[test_ind]

    return train_data, test_data


def generate_network(train, layers, inDrop, drop):
    # fill the architecture params from dict
    dsn1_layers = layers["DSN_1"].split("-")
    dsn2_layers = layers["DSN_2"].split("-")
    snp_layers = layers["SPN"].split("-")
    # contruct two parallel networks
    for l in range(len(dsn1_layers)):
        if l == 0:
            input_drug1 = Input(shape=(train["drug1"].shape[1],))
            middle_layer = Dense(int(dsn1_layers[l]), activation='relu', kernel_initializer='he_normal')(input_drug1)
            middle_layer = Dropout(float(inDrop))(middle_layer)
        elif l == (len(dsn1_layers) - 1):
            dsn1_output = Dense(int(dsn1_layers[l]), activation='linear')(middle_layer)
        else:
            middle_layer = Dense(int(dsn1_layers[l]), activation='relu')(middle_layer)
            middle_layer = Dropout(float(drop))(middle_layer)

    for l in range(len(dsn2_layers)):
        if l == 0:
            input_drug2 = Input(shape=(train["drug2"].shape[1],))
            middle_layer = Dense(int(dsn2_layers[l]), activation='relu', kernel_initializer='he_normal')(input_drug2)
            middle_layer = Dropout(float(inDrop))(middle_layer)
        elif l == (len(dsn2_layers) - 1):
            dsn2_output = Dense(int(dsn2_layers[l]), activation='linear')(middle_layer)
        else:
            middle_layer = Dense(int(dsn2_layers[l]), activation='relu')(middle_layer)
            middle_layer = Dropout(float(drop))(middle_layer)

    concatModel = concatenate([dsn1_output, dsn2_output])

    for snp_layer in range(len(snp_layers)):
        if len(snp_layers) == 1:
            snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(concatModel)
            snp_output = Dense(1, activation='linear')(snpFC)
        else:
            # more than one FC layer at concat
            if snp_layer == 0:
                snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(concatModel)
                snpFC = Dropout(float(drop))(snpFC)
            elif snp_layer == (len(snp_layers) - 1):
                snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(snpFC)
                snp_output = Dense(1, activation='linear')(snpFC)
            else:
                snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(snpFC)
                snpFC = Dropout(float(drop))(snpFC)

    model = Model([input_drug1, input_drug2], snp_output)
    return model


def trainer_no_val(model, l_rate, train_data, max_epoch, batch_size, model_name, loss_weight):
    print("Training model without validation set")
    model.compile(optimizer=keras.optimizers.Adam(lr=float(l_rate), beta_1=0.9, beta_2=0.999, amsgrad=False),
                  loss='mean_squared_error',
                  loss_weights=loss_weight)

    checkpoint = ModelCheckpoint(model_name, monitor='loss', save_best_only=True, mode='min', verbose=1)
    history = model.fit([train_data['drug1'], train_data['drug2']],
                        train_data['y'], batch_size=batch_size,
                        epochs=max_epoch, shuffle=True,
                        callbacks=[checkpoint], verbose=1)
    return model

def predict(model, data):
    pred = model.predict(data)
    return pred.flatten()
