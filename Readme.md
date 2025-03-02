# README

---
‘Improving drug combination prediction based on imputation of gene expression profiles induced by varying drug doses’  



##Requirements
- python==3.11.5
- pytorch==2.0.1
- tensorflow==2.15
- numpy==1.25.2
- scipy==1.12.0
- pandas==2.1.1
- scikit-learn=1.2.2
- kears=2.15.0
- shap==0.46.0
- pypots==0.8.1


##Instruction
The project focuses on enhancing drug combination prediction through gene expression profile imputation, using a series of models and datasets.  



##Repository Structure
###Data
`data/complete-dose.csv`: This file contains the complete - dose dataset constructed in the paper. It serves as the foundation for training the BRIDS model.  

`data/Pre-impute.csv`: Data before imputation.

`data/Post-impute.csv`: Data after imputation.

###Code
`code/impute`: This directory includes 25 classic sequence prediction imputation models. The BRIDS model used in the paper is also here, and the other models serve as comparison methods. After training, you can use `Load_BRIDS2impute.py` to load the trained BRIDS model and predict the real missing data to obtain the imputed data.  

###Prediction Models
`code/predict`: This folder contains five classic combination drug synergy prediction models and the random forest prediction model (named 'My') proposed in this paper. For the specific execution methods of these models, please refer to the `readme.md` files in each sub-folder.  


##Usage
###Imputation
- Train the BRIDS model using the `data/complete-dose.csv` dataset.  
- After training, run `Load_BRIDS2impute.py` to load the model and perform imputation on the real missing data. You can find the imputed data in `data/Post-impute.csv`.  


###Drug Combination Prediction
Apply the imputed data to the models in the `predict` folder. Follow the instructions in the respective `readme.md` files for each model to execute the drug combination prediction tasks.

###Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com).
