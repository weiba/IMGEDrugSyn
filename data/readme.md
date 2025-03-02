#README for L1000 Gene Expression Datasets

This repository contains three files related to gene expression data used in our study for drug combination prediction models. Below is a detailed description of the data sources, total data size, and the meanings of rows and columns in the datasets.
##1. Data Source
All data in these files are sourced from the L1000 dataset, which provides gene expression profiles (GP) in response to chemical perturbations (drug treatments) across different doses. This data is used for modeling drug combination responses, and we used these data to train our BRIDS model for imputing missing values in dose-dependent gene expression profiles.

##2. Data Summary
- complete-dose.csv

	- This file contains the complete-dose GP dataset, which is used as the foundation for training the BRIDS model. It includes gene expression data across four drug doses for various drugs and cell lines.
	- The dataset contains 10,520 entries in total.Specifically, this dataset includes 22 cell lines and 869 drugs, with each drug having four dose points.
	- The data consists of 2,630 complete GP profiles, with each profile representing the gene expression response to a specific drug-cell line combination at all four dose points (0.08 µM, 0.4 µM, 2 µM, 10 µM).
- Pre-impute.csv

	- This file contains the incomplete-dose GP dataset before imputation.
	- It includes 68 drugs and 16 cell lines, yielding a total of 418 data entries.
Each entry corresponds to a drug's gene expression data at one or more doses in a given cell line. Some dose points may be missing in this dataset, which is why imputation is required.
- Post-impute.csv

- This file contains the imputed GP dataset, which is the result after applying the BRIDS model to impute the missing values in the Pre-impute.csv dataset.
	- The dataset consists of 68 drugs and 16 cell lines, similar to the Pre-impute.csv, but each drug-cell line combination now has gene expression data for all four doses.After imputation, the dataset contains 1,672 data entries in total, with each entry having 4 × 977 gene expression features (for 4 doses and 977 gene expression features).

##3. Row Meaning
Each row in the dataset represents a gene. Genes are identified using Entrez Gene IDs, which are unique identifiers provided by NCBI (National Center for Biotechnology Information) for genes, proteins, and other biological entities.

##4. Column Meaning
- Perturbation IDs (pert_id):


The columns in the dataset are labeled using pert_id, which refers to the drug-cell line treatment condition. For example, in a column like LJP001_SKBR3_6H:BRD-K68065987-300-06-7:10, the components can be interpreted as:

- SKBR3: The specific cell line used (in this case, SKBR3).
- 6H: Indicates the time point of gene expression measurement after drug treatment (6 hours).
- BRD-K68065987: The drug identifier (unique identifier for the drug being used in this treatment).
- 10: Represents the drug dose used, in this case, 10 µM.


The pert_id therefore contains information on the cell line, drug, experimental conditions, and dose point.