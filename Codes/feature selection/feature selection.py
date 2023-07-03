# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:04:14 2021


import csv,pandas as pd, numpy as np
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

filename = 'dataset.csv'
dataset = pd.read_csv(filename)
data = np.array(dataset)
X = data[:, 3:26]
Y = data[:, 2]
normer = StandardScaler()
data = normer.fit_transform(X)

validation_size = 0.2
seed = np.random.randint(0, 999999)
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = validation_size, random_state = seed)
trainingset = pd.DataFrame(np.concatenate([y_train.reshape(-1, 1), x_train], axis=1), columns=dataset.columns[2:])
testset = pd.DataFrame(np.concatenate([y_test.reshape(-1, 1), x_test], axis=1), columns=dataset.columns[2:])
trainingset.to_excel("splited_trainingset.xlsx")
testset.to_excel("splited_testset.xlsx")

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor

models={}
models['LR']= LinearRegression
models['LASSO']=Lasso
models['EN']= ElasticNet
models['DTR']= DecisionTreeRegressor
models['KNR']= KNeighborsRegressor
models['SVR']= SVR
models['RFR']= RandomForestRegressor
models['GBR']= GradientBoostingRegressor
models['ETR']= ExtraTreesRegressor
models['ABR']= AdaBoostRegressor
# results = []
# for key in models:
#     kfold = KFold(n_splits=90, random_state=seed, shuffle=True)
#     scoring = 'neg_mean_squared_error'
#     cv_result = cross_val_score(models[key],x_train,y_train,cv=kfold,scoring = scoring)
#     results.append(cv_result)
#     print ('%s: %f'% (key, cv_result.mean()))

# for name, algo in models.items():
#     kfold = KFold(n_splits=90, shuffle=True, random_state=seed)
#     cv_preds = []
#     cv_obs = []
#     for itrain, itest in kfold.split(x_train):
        
#         model = algo()
#         model.fit(x_train[itrain], y_train[itrain])  
#         cv_obs.append(y_train[itest])
#         cv_preds.append(model.predict(x_train[itest, :]))
    
#     cv_obs = np.concatenate(cv_obs)
#     cv_preds = np.concatenate(cv_preds)
#     print("="*10)
#     print(f"{name}")
#     print(f"cv r2: {round(r2_score(cv_obs, cv_preds), 2)}")
#     print(f"cv mse: {round(mean_squared_error(cv_obs, cv_preds), 2)}")
    
#     model = algo()
#     model.fit(x_train, y_train)
#     test_pred = model.predict(x_test)
#     train_pred = model.predict(x_train)
#     print(f"train r2: {round(r2_score(y_train, train_pred), 2)}")
#     print(f"train mse: {round(mean_squared_error(y_train, train_pred), 2)}")
#     print(f"test r2: {round(r2_score(y_test, test_pred), 2)}")
#     print(f"test mse: {round(mean_squared_error(y_test, test_pred), 2)}")
    
# from genetic_selection import GeneticSelectionCV
# estimator = RandomForestRegressor()
# selector = GeneticSelectionCV(estimator,
#                                   cv=10,
#                                   verbose=1,
#                                   scoring="neg_mean_squared_error",
#                                   max_features=21,
#                                   n_population=50,
#                                   crossover_proba=0.5,
#                                   mutation_proba=0.2,
#                                   n_generations=40,
#                                   crossover_independent_proba=0.5,
#                                   mutation_independent_proba=0.05,
#                                   tournament_size=3,
#                                   n_gen_no_change=10,
#                                   caching=True,
#                                   n_jobs=1)
# selector = selector.fit(x_train,y_train)
# print(selector.support_)
# print(np.array(selector.population_).shape)
# print(selector.generation_scores_)
# x_train, x_test = x_train[:,selector.support_], x_test[:,selector.support_]
# estimator.fit(x_train,y_train)
# y_test_pred = estimator.predoct(x_test)
# r2_test = r2_score(y_test,y_test_pred)
# mse_test = mean_squared_error(y_test,y_test_pred)
# print(r2_test,mse_test)

# pip install fast-machine-learning

from fml.data import DataObject
from fml.pipelines import SHAPModelling
from fml.validates import Validate

trainobj = DataObject(X=x_train, Y=y_train)
testobj = DataObject(X=x_test, Y=y_test)

feature_importances = dict()
results = {}
for name, algo in models.items():
    s = SHAPModelling().fit(algo, trainobj, testobj)
    result = []
    for max_f in range(1, 22 ):
        new_trainobj, new_testobj = s.transform(max_f)
        v = Validate(algo, new_trainobj, new_testobj)
        v.validate_loo()
        v.validate_test()
        loo_r = v.loo_result
        test_r = v.test_result
        
        result.append([max_f, loo_r['r2_score'], loo_r['rmse'], test_r['r2_score'], test_r['rmse']])
        print (f"{name},{max_f}")
        # print(f"max_f: {max_f}")
        # print(f"loo R2: {round(loo_r['r2_score'], 2)}, loo rmse: {round(loo_r['rmse'], 2)}")
        # print(f"test R2: {round(test_r['r2_score'], 2)}, test rmse: {round(test_r['rmse'], 2)}")
    feature_importance = s.feature_selection.feature_shap
    result = pd.DataFrame(result, columns=["max_f", "loo r2", "loo rmse", "test r2", "test rmse"])
    
    results[name] = [result, feature_importance]
    
    feature_importances[name] = feature_importance
    
    
# selected_algos = ["SVR", "LR"]
selected_algos = models.keys()
for selected_algo in selected_algos:
    fi = feature_importances[selected_algo][:, 0].astype(int)
    
    selected_trainingset = trainingset.iloc[:, 1:].iloc[:, fi]
    selected_trainingset = pd.concat([trainingset.iloc[:, 0], selected_trainingset], axis=1)
    selected_testset = testset.iloc[:, 1:].iloc[:, fi]
    selected_testset = pd.concat([testset.iloc[:, 0], selected_testset], axis=1)
    
    selected_trainingset.to_excel(f"{selected_algo}-trainingset.xlsx")
    selected_testset.to_excel(f"{selected_algo}-testset.xlsx")


