from model.random_forest import predict


df = get_data(settings.DATA.data_set)
X_train, X_test, y_train, y_test = split_data(df) 
clf = train_decision_tree(X_train, y_train)

logging.info(f'Accuracy is {clf.score(X_test, y_test)  } ')

responce = predict(X_test, path_to_model)


logging.info(f'Prediction is {clf .predict(X_test )  } ')