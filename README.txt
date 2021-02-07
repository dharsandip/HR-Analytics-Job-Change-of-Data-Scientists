This project is based on "HR Analytics: Job Change of Data Scientists, Predict who will move to a new job" Dataset from Kaggle. The whole data was divided to train and test set. The dataset is quite imbalanced. we had to take care of that.
Most of the features are categorical, some with high cardinality. Here, at first we are doing all the necessary data preprocessing steps, model building, predictions, anslysis of results etc. in research environment (Jupyter notebook) interactively.
Later we wrote production level code in python for the machine learning pipeline. Missing imputation is a part of our pipeline along with other necessary steps. We are using Scikit Learn Pipeline here for production code. We made sure that the predictions obtained in the Research Environment matched with the predictions that we got by wriring and running production level code of the machine learning pipeline. 
In order to take care of the imbalanced training data, we are using resampling technique (hybrid resampling -> combination Over-Sampling (using SMOTE) and Under-Sampling (using RandomUnderSampler)). This improved results quite a lot. 
We tried Random Forest and XGBoost model both and later selled with XGBoost model since it gave better predictions.


Various columns of training data set:
enrollee_id : Unique ID for candidate
city: City code
city_ development _index : Developement index of the city (scaled)
gender: Gender of candidate
relevent_experience: Relevant experience of candidate
enrolled_university: Type of University course enrolled if any
education_level: Education level of candidate
major_discipline :Education major discipline of candidate
experience: Candidate total experience in years
company_size: No of employees in current employer's company
company_type : Type of current employer
lastnewjob: Difference in years between previous job and current job
training_hours: training hours completed
target: 0 – Not looking for job change, 1 – Looking for a job change

Results:

With Resampling of training dataset and XGBoost model:

Classification report of Test Set:

precision    recall  f1-score   support

         0.0       0.89      0.80      0.84      1553
         1.0       0.57      0.72      0.64       576

    accuracy                           0.78      2129
   macro avg       0.73      0.76      0.74      2129
weighted avg       0.80      0.78      0.79      2129

roc_auc_score for the test set:  0.761528426164413


Classification report of Training Set:

precision    recall  f1-score   support

         0.0       0.90      0.80      0.85     14380
         1.0       0.75      0.87      0.81     10066

    accuracy                           0.83     24446
   macro avg       0.83      0.84      0.83     24446
weighted avg       0.84      0.83      0.83     24446

roc_auc_score for the training set:  0.8371100735148024






