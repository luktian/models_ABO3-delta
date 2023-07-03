
from fml.data import read_data, DataObject
from sklearn.ensemble import RandomForestRegressor
import numpy as np, pandas as pd
from fml.validates import Validate

algo = RandomForestRegressor

train = DataObject().from_df(read_data("./train set.csv").astype(float))
test = DataObject().from_df(read_data("./test set.csv").astype(float))

v = Validate(algo, train, test, **dict(n_estimators=10,random_state = 2))
v.validate_all()
results = v.results

loo_values = np.concatenate([results["loo"]["true_value"].reshape(-1, 1), results["loo"]["preds"].reshape(-1, 1)], axis=1)
test_values = np.concatenate([results["test"]["true_value"].reshape(-1, 1), results["test"]["preds"].reshape(-1, 1)], axis=1)
train_values = np.concatenate([results["train"]["true_value"].reshape(-1, 1), results["train"]["preds"].reshape(-1, 1)], axis=1)
loo_values = pd.DataFrame(loo_values, columns=["obs","pred"])
loo_values.index = train.indexes
test_values = pd.DataFrame(test_values, columns=["obs","pred"])
test_values.index = test.indexes
train_values = pd.DataFrame(train_values, columns=["obs","pred"])
train_values.index = train.indexes

train_values.to_excel(f"{algo.__name__}-train-values.xlsx")
test_values.to_excel(f"{algo.__name__}-test-values.xlsx")
loo_values.to_excel(f"{algo.__name__}-loo-values.xlsx")


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
result_table.to_excel(f"{algo.__name__}-result-table.xlsx")
