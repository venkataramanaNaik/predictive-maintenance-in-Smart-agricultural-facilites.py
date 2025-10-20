import xgboost as xgb
from xgboost import DMatrix

dtrain = DMatrix(X_train, label=y_train)
dtest = DMatrix(X_test, label=y_test)
params = {
    'max_depth': 6,
    'eta': 0.3,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

xgb_model = xgb.train(params, dtrain, num_boost_round=100)
predictions = xgb_model.predict(dtest)
