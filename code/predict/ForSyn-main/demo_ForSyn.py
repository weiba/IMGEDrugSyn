from gcForest import gcForest
from logger import get_logger
from sklearn.model_selection import train_test_split,StratifiedKFold,RepeatedStratifiedKFold
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from evaluation import accuracy,f1_binary,f1_macro,f1_micro
from load_data import load_data
from feature_selection import select_feature
from sklearn.metrics import average_precision_score, matthews_corrcoef, f1_score, recall_score, confusion_matrix, \
    classification_report, roc_auc_score, auc, precision_recall_curve, accuracy_score, classification_report, \
    precision_score
from imblearn.ensemble import BalancedBaggingClassifier,RUSBoostClassifier,BalancedRandomForestClassifier
from imblearn.metrics import geometric_mean_score

x,y,dataid=load_data()
#x(3182,1956)特征    y(3192) 标签 123 还原就是012 只有1是阳性，01为阴性
for index in range(len(y)):
    if y[index]==3.0:
        y[index]=0.0    
    if y[index]==1.0:
        y[index]=0.0
    if y[index]==2.0:
        y[index]=1.0
print("x_shape:",x.shape)
print("y_shape:",y.shape)
print("y_distribution:",Counter(y)) 

def get_config():
    config={}
    config["random_state"]=None
    config["max_layers"]=100
    config["early_stop_rounds"]=1
    config["if_stacking"]=False
    config["if_save_model"]=False
    config["train_evaluation"]=f1_macro ##f1_binary,f1_macro,f1_micro,accuracy
    config["estimator_configs"]=[]
    # for i in range(10):
    #     config["estimator_configs"].append({"n_fold":5,"type":"IMRF","n_estimators":40,"splitter":"best"})
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":100,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":100,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":100,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":100,"n_jobs":-1})
    config["output_layer_config"]=[]
    return config

if __name__=="__main__":   
    config=get_config()  
    skf=RepeatedStratifiedKFold(n_splits=5,random_state=33,n_repeats=1)

    f1s=[]
    auprs=[]
    aucs = []  # For storing AUC scores
    mccs=[]
    recalls=[]
    gmeans=[]
    precisions=[]
    accuracies=[]
    i=1
    for train_id,test_id in skf.split(x,y):
        print("============{}-th cross validation============".format(i))
        x_train,x_test,y_train,y_test=x[train_id],x[test_id],y[train_id],y[test_id]
        index=select_feature(x_train,y_train, 1806)#1806
        x_train=x_train[:,index]
        x_test=x_test[:,index]
        config=get_config()
        gc=gcForest(config)
        gc.fit(x_train,y_train)
        y_pred=gc.predict(x_test)
        
        #calculate y_score        
        y_pred_prob=gc.predict_proba(x_test)
        y_score=[]
        for item in y_pred_prob:
            y_score.append(item[1])
        y_score=np.array(y_score)
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        aupr = auc(recall,precision)
        # Calculate ROC AUC
        auc_score = roc_auc_score(y_test, y_score)
                  
        f1 = f1_score(y_test, y_pred,average='binary')       
        recall = recall_score(y_test, y_pred,average="binary")
        mcc = matthews_corrcoef(y_test, y_pred)
        gmean= geometric_mean_score(y_test, y_pred,average='binary')
        precision = precision_score(y_test, y_pred, average='binary')
        accuracy = accuracy_score(y_test, y_pred)
          
        f1s.append(f1)
        auprs.append(aupr)
        aucs.append(auc_score)  # Append AUC score
        recalls.append(recall)
        mccs.append(mcc)
        gmeans.append(gmean)
        precisions.append(precision)
        accuracies.append(accuracy)
        i+=1
        
    print("============training finished============")
    f1s=np.array(f1s)
    auprs=np.array(auprs)
    recalls=np.array(recalls)
    mccs=np.array(mccs)
    gmeans=np.array(gmeans)
    precisions=np.array(precisions)
    accuracies=np.array(accuracies)

    print("f1 ", f1s)
    print("aupr ", auprs)
    print("auc ", aucs)  # Print AUC average
    print("recall ", recalls)
    print("mcc :", mccs)
    print("gmean :", gmeans)
    print(f"Precision : {precisions}")
    print(f"Accuracy : {accuracies}")

    print('-------------------')
    print("Data:",dataid)
    print("Model: ForSyn")
    print("f1 average:", np.mean(f1s))
    print("aupr average:", np.mean(auprs))
    print("auc average:", np.mean(aucs))  # Print AUC average
    print("recall average:", np.mean(recalls))
    print("mcc average:", np.mean(mccs))
    print("gmean average:", np.mean(gmeans))
    print(f"Precision average: {np.mean(precisions)}")
    print(f"Accuracy average: {np.mean(accuracies)}")

    # Calculate and print the variance for each metric
    print('-------------------')
    print("f1 variance:", np.std(f1s))
    print("aupr variance:", np.std(auprs))
    print("auc variance:", np.std(aucs))
    print("recall variance:", np.std(recalls))
    print("mcc variance:", np.std(mccs))
    print("gmean variance:", np.std(gmeans))
    print(f"Precision variance: {np.std(precisions)}")
    print(f"Accuracy variance: {np.std(accuracies)}")