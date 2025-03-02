MatchMaker Five - Fold Cross - Validation Code   

This repository contains the code for performing MatchMaker five - fold cross - validation. The code can be directly executed, and you can easily adjust the input to use different feature combinations.
Usage  

You can run the code by executing the following command in your terminal:

    python main_fold_noval.py
  
Feature Combinations  
The code supports multiple feature combinations. You can simply change the input data files to use different combinations:  
data1.csv: Imputed data for drug 1.  
data1_977.csv: Data for drug 1 before imputation.  
data1_633_ChemoPy.csv: Original features.  
data1_1610_ChemoPy+GPno.csv: Original features + data before imputation.
data1_4541_ChemoPy+GP.csv: Original features + imputed data.