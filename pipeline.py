from sklearn import xgboost

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

import preprocessors as pp
import config


# Creating pipeline for the data preprocessing steps
analytics_pipe1 = Pipeline(
    [
        ('categorical_imputer',
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA)),
         
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES)),
         
        ('scaler', MinMaxScaler()),
    ]
)

# Creating pipeline for the machine learning model
analytics_pipe2 = Pipeline(
    [
        
        ('xgboost_classification', xgboost.XGBClassifier(n_estimators=550, scale_pos_weight=1.9))
        
    ]
)