from model.random_forest import predict_rf
from model.naive_bayes import predict_naive
from conf.conf import settings


predict_naive([settings.PREDICTION.data_predict])

# df = get_data(settings.DATA.data_set)
# X_train, X_test, y_train, y_test = split_data(df) 
# clf = train_decision_tree(X_train, y_train)

# logging.info(f'Accuracy is {clf.score(X_test, y_test)  } ')

# responce = predict(X_test, path_to_model)


# logging.info(f'Prediction is {clf .predict(X_test )  } ')