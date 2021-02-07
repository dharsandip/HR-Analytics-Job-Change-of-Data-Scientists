# data
TRAINING_DATA_FILE = "aug_train.csv"
TEST_DATA_FILE = "aug_test.csv"
PIPELINE1_NAME = 'data_preprocessing'
PIPELINE2_NAME = 'xgboost_classification'

TARGET = 'target'

# input variables 
FEATURES = ['enrollee_id', 'city', 'city_development_index', 'gender', 'relevent_experience',
            'enrolled_university', 'education_level', 'major_discipline',
            'experience', 'company_size', 'company_type', 'company_type',
            'last_new_job', 'training_hours']

# Drop features
DROP_FEATURES = ['enrollee_id', 'city']


# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = ['gender', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job']



# categorical variables to encode
CATEGORICAL_VARS = ['gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job']

