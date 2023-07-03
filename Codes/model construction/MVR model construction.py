
from fml.data import read_data, DataObject
from fml.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
import numpy as np, pandas as pd


dir_path = ["./ETR", "./GBR", "./RFR"]

trains = [ DataObject().from_df(read_data(f"{i}./train set.csv").astype(float)) for i in dir_path ]
tests = [ DataObject().from_df(read_data(f"{i}./test set.csv").astype(float)) for i in dir_path ]

algos = [GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor]

model_ps = [
    {"n_estimators": 11}, 
    {"n_estimators": 99},
    {"n_estimators": 10, "random_state": 2}
    ]

vr = VotingRegressor(verbose=True, rounds=100)
vr.fit(algos, trains, tests, model_ps)
results = vr.results
best_w = vr.best_weights

loo_values = np.concatenate([results["loo"]["true_value"].reshape(-1, 1), results["loo"]["preds"].reshape(-1, 1)], axis=1)
test_values = np.concatenate([results["test"]["true_value"].reshape(-1, 1), results["test"]["preds"].reshape(-1, 1)], axis=1)
train_values = np.concatenate([results["train"]["true_value"].reshape(-1, 1), results["train"]["preds"].reshape(-1, 1)], axis=1)
loo_values = pd.DataFrame(loo_values, columns=["obs","pred"])
loo_values.index = trains[0].indexes
test_values = pd.DataFrame(test_values, columns=["obs","pred"])
test_values.index = tests[0].indexes
train_values = pd.DataFrame(train_values, columns=["obs","pred"])
train_values.index = trains[0].indexes

train_values.to_excel("meta-train-values.xlsx")
test_values.to_excel("meta-test-values.xlsx")
loo_values.to_excel("meta-loo-values.xlsx")


result_keys = ["mae", "mse", "R", "r2_score", "rmse"]
result_index = []
result_table = []
for i, j in results.items():
    tmp = []
    for result_key in result_keys:
        tmp.append(j[result_key])
    result_table.append(tmp)
    result_index.append(i)
result_table = pd.DataFrame(result_table, columns=result_keys, index=result_index)
result_table.to_excel("meta-result_table.xlsx")
