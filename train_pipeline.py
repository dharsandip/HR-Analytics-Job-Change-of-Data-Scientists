import numpy as np
import pandas as pd

import joblib

import pipeline
import config
from collections import Counter
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


def run_training():
    """Train the model."""

    # read training data
    data_train = pd.read_csv(config.TRAINING_DATA_FILE)
    data_test = pd.read_csv(config.TEST_DATA_FILE)

    X_train = data_train[config.FEATURES]
    y_train = data_train[config.TARGET]
    X_test = data_test[config.FEATURES]
    y_test = np.load("jobchange_test_target_values.npy")
 
# Data Preprocessing    
    pipeline.analytics_pipe1.fit(X_train, y_train)
 
# Saving the pipeline of Data Preprocessing steps    
    joblib.dump(pipeline.analytics_pipe1, config.PIPELINE1_NAME)
    
    print("Before Resampling of training data: ")
    counter = Counter(y_train)
    print(counter)
    
    _pipe_analytics1 = joblib.load(filename=config.PIPELINE1_NAME)
    
    data_train = pd.read_csv(config.TRAINING_DATA_FILE)
    data_test = pd.read_csv(config.TEST_DATA_FILE)
    X_train = data_train[config.FEATURES]
    y_train = data_train[config.TARGET]
    X_test = data_test[config.FEATURES]
    y_test = np.load("jobchange_test_target_values.npy")
     
    X_train = _pipe_analytics1.fit_transform(X_train)
 
# Applying hybrid resampling (combination of over sampling and under sampling) on the training dataset    
    over = SMOTE(sampling_strategy=0.7)
    under = RandomUnderSampler(sampling_strategy=0.7)
    steps = [('o', over), ('u', under)]
    pipeline1 = Pipeline(steps=steps)	
    X_train, y_train = pipeline1.fit_resample(X_train, y_train) 
    
    print("After Resampling of training data: ")
    counter = Counter(y_train)
    print(counter)
    
    pipeline.analytics_pipe2.fit(X_train, y_train)

# Saving the pipeline of Model    
    joblib.dump(pipeline.analytics_pipe2, config.PIPELINE2_NAME)

# Predicting the target values of training set    
    y_pred_train = pipeline.analytics_pipe2.predict(X_train)

# determine classification_report and roc_auc_score for the training set  
    print("Classification report for the Training Set:")
    print()
    print(classification_report(y_train, y_pred_train))
    print("roc_auc_score for the Training Set: {}".format(roc_auc_score(y_train, y_pred_train)))
    print()
    
    print("Training is completed")
    

if __name__ == '__main__':
    run_training()

