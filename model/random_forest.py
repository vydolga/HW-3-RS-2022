from connector.connector import get_data
from conf.conf import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from util.util import  save_model, load_model
from conf.conf import settings
# from conf.config import settings

def split_data(df):

    logging.info('Defining X and y')
    # Filter out target column
    X = df.iloc[:, :-1]
    
    # Select target column
    y = df['target']
    
    logging.info('Splitting dataset')
    #split variables into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, #independent variables
                                                        y, #dependent variable
                                                        random_state = 3)
    return X_train, X_test, y_train, y_test                                                 

def train_random_forest(X_train, y_train):
    # Initialize the model
    clf = RandomForestClassifier(max_depth = 3,
                                random_state = 10
                                )
    logging.info('Training the model ')

    #train the model
    clf.fit(X_train, y_train)
    save_model(dir = settings.MODEL.dt_random, model = clf)
    return clf

def predict_rf(values):
    clf = load_model(settings.MODEL.dt_random)
    return clf.predict(values)

df = get_data(settings.DATA.data_set)
X_train, X_test, y_train, y_test = split_data(df) 
clf = train_random_forest(X_train, y_train)

logging.info(f'Accuracy is {clf.score(X_test, y_test)  } ')

responce = predict_rf(X_test)


logging.info(f'Prediction is {clf .predict(X_test )  } ')
