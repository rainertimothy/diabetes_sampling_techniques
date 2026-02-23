# diabetes_sampling_techniques
Source code of this paper => Comparative Analysis of Machine Learning Models for Early Diabetes Prediction based on Imbalance Data Handling Techniques (link soon). This paper was published in IEEE access
Dataset => DiaBD: A diabetes dataset for enhanced risk analysis and research in Bangladesh (https://data.mendeley.com/datasets/m8cgwxs9s6/3)

This code consists of three parts, which are parts of research framework. 

1. Samplings part, which is highlights of the research. In this part, i use 3 methods of sampling: undersampling, oversampling, and sampling ratio. Each methods have their own variability of techniques. Act as a baseline, non-sampling or original data without sampling also get tested
Undersampling: RandomUnderSampler
Oversampling: ADASYN, SMOTE, SMOTETomek
Sampling ratio: 1:2, 1:3, 1:4, 1:5, 1:10 

2. Hyperparameter tuning, this code use randomizedsearchcv as hyperparameter tuning choice, the reason is this parameter could operate in high speed and cost-less memory. Code also use cross-validation, to validate the data tested in training process, which divide data into 5-different folds. Five-fold cross-validation was choosen, as it returns optimal performance when tested in several metrics

3. Metrics, code use four metrics, which is accuracy, precision, recall, and f1-score to measure model performance
