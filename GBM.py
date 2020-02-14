#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:37:40 2020

@author: weiya
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from typing import Tuple, Dict, List
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# XGBoost
import xgboost as xgb
import pandas as pd

from sklearn.model_selection import train_test_split

# glmnet
import glmnet_python
from glmnet import glmnet
from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
from glmnetPlot import glmnetPlot
from cvglmnetPlot import cvglmnetPlot
from cvglmnetPredict import cvglmnetPredict

def calc_accuracy(pred, truth):
    specificity = sum((1 - pred) * (1 - truth))/ sum(1 - truth)
    sensitivity = sum(pred * truth ) / sum(truth)
    overall = sum(pred == truth) / len(truth)
    return specificity, sensitivity, overall

# load datasets
# folder = "BrainCancer/Data"
folder = "data"
survival_outcome1 = np.loadtxt(f"{folder}/sc1_Phase1_GE_Outcome.tsv", skiprows=1, usecols=1, dtype="int")
expression = pd.read_csv(f"{folder}/sc1_Phase1_GE_FeatureMatrix.tsv", header = 0, index_col = 0, sep = "\t")
phenotype1 = pd.read_csv(f"{folder}/sc1_Phase1_GE_Phenotype.tsv", header = 0, index_col = 0, sep = "\t", na_values = [" ", ""])

survival_outcome2 = np.loadtxt(f"{folder}/sc2_Phase1_CN_Outcome.tsv", skiprows=1, usecols=1, dtype="int")
copy_number = pd.read_csv(f"{folder}/sc2_Phase1_CN_FeatureMatrix.tsv", header = 0, index_col = 0, sep = "\t")
phenotype2 = pd.read_csv(f"{folder}/sc2_Phase1_CN_Phenotype.tsv", header = 0, index_col = 0, sep = "\t", na_values = [" ", ""])

survival_outcome3 = np.loadtxt(f"{folder}/sc3_Phase1_CN_GE_Outcome.tsv", skiprows=1, usecols=1)
cn_ge = pd.read_csv(f"{folder}/sc3_Phase1_CN_GE_FeatureMatrix.tsv", header = 0, index_col = 0, sep = "\t")
phenotype3 = pd.read_csv(f"{folder}/sc3_Phase1_CN_GE_Phenotype.tsv", header = 0, index_col = 0, sep = "\t", na_values = [" ", ""])

#sc1 = pd.concat([expression, phenotype1], axis = 1)
#encoded_sc1 = pd.get_dummies(sc1)
dummy_phenotype1 = pd.get_dummies(phenotype1)
sc1 = pd.concat([expression, dummy_phenotype1], axis = 1)

#sc2 = pd.concat([copy_number, phenotype2], axis = 1)
#encoded_sc2 = pd.get_dummies(sc2)
dummy_phenotype2 = pd.get_dummies(phenotype2)
sc2 = pd.concat([copy_number, dummy_phenotype2], axis = 1)

    
X_train_sc1, X_test_sc1, y_train_sc1, y_test_sc1 = train_test_split(sc1, survival_outcome1, random_state = 0, test_size = 0.2)
X_train_sc2, X_test_sc2, y_train_sc2, y_test_sc2 = train_test_split(sc2, survival_outcome2, random_state = 0, test_size = 0.2)


# ----------------------------------------------------
# sub-challenge 1
# ----------------------------------------------------

# dtrain = xgb.DMatrix(X_train_sc1, label = y_train_sc1)
# dtest = xgb.DMatrix(X_test_sc1, label = y_test_sc1)

# fit by logistic with lasso

#cvfit = cvglmnet(x = X_train_sc1.values.copy(), y = y_train_sc1.astype("float").copy(), family = "binomial", alpha = 1)
cvfit = cvglmnet(x = X_train_sc1.values.copy(), y = y_train_sc1.astype("float").copy(), family = "binomial", alpha = 1, weights = (y_train_sc1.reshape(-1, 1) == 0) * 5.2 + 1)
preds = cvglmnetPredict(cvfit, X_test_sc1.values, s = 'lambda_min', ptype = "class").reshape(1, -1)[0]
coef = cvglmnetCoef(cvfit, s = 'lambda_min').reshape(1, -1)[0]
coef_1se_left = cvglmnetCoef(cvfit, s = np.exp(np.log(cvfit["lambda_min"]) * 2 - np.log(cvfit["lambda_1se"]))).reshape(1, -1)[0]
lasso_idx = np.nonzero(coef_1se_left)[0][1:]

dtrain_lasso = xgb.DMatrix(X_train_sc1.values[:, lasso_idx - 1], label = y_train_sc1)
dtest_lasso = xgb.DMatrix(X_test_sc1.values[:, lasso_idx - 1], label = y_test_sc1)
evallist_lasso = [(dtrain_lasso, 'train'), (dtest_lasso, 'eval')]
param = {'max_depth':6, 'eta':0.001, 'objective':'binary:logistic', 'subsample': 0.3, 
         'scale_pos_weight': 0.15, 
#         'max_delta_step': 1,
 #        'gamma': 0.5
         }
#num_round = 1000000
num_round = 10000
param["eval_metric"] = "auc" 
#bst = xgb.train(param, dtrain_lasso, num_round, evals = evallist_lasso, early_stopping_rounds = 1000)
bst = xgb.train(param, dtrain_lasso, num_round, evals = evallist_lasso)
#num_round = 4000
#bst = xgb.train(param, dtrain_lasso, num_round, evals = evallist_lasso)
preds = bst.predict(dtest_lasso)
print(calc_accuracy( (preds > 0.5) * 1, y_test_sc1))
confusion_matrix((preds > 0.5) * 1, y_test_sc1)

# get feature names

sorted_features_by_xgb = [k for k, v in sorted(bst.get_fscore().items(), key = lambda item: item[1])]
# top 20 predictors
#for fname in sorted_features_by_xgb[::-1][0:20]:
for fname in sorted_features_by_xgb[::-1]:
    idx = int(fname.split("f")[1])
    print(X_train_sc1.columns.values[lasso_idx[idx] - 1])
    
features_selected_from_sc1 = []
for fname in sorted_features_by_xgb[::-1]:
    features_selected_from_sc1.append(lasso_idx[int(fname.split("f")[1])] - 1)

# ----------------------------------------------------
# sub-challenge 2
# ----------------------------------------------------

# dtrain2 = xgb.DMatrix(X_train_sc2, label = y_train_sc2)
# dtest2 = xgb.DMatrix(X_test_sc2, label = y_test_sc2)

cvfit2 = cvglmnet(x = X_train_sc2.values.copy(), y = y_train_sc2.astype("float").copy(), family = "binomial", alpha = 1)#, weights = y_train_sc2.reshape((-1,1))*1.0 + 1)
preds2 = cvglmnetPredict(cvfit2, X_test_sc2.values, s = 'lambda_min', ptype = "class").reshape(1, -1)[0]

coef2 = cvglmnetCoef(cvfit2, s = 'lambda_min').reshape(1, -1)[0]
coef2_left = cvglmnetCoef(cvfit2, s = np.array([np.exp(-3)])).reshape(1, -1)[0]
lasso_idx2 = np.nonzero(coef2_left)[0][1:] # the first one corresponds to the intercepts

dtrain2_lasso = xgb.DMatrix(X_train_sc2.values[:, lasso_idx2 - 1], label = y_train_sc2)
dtest2_lasso = xgb.DMatrix(X_test_sc2.values[:, lasso_idx2 - 1], label = y_test_sc2)
evallist2_lasso = [(dtrain2_lasso, 'train'), (dtest2_lasso, 'eval')]
param = {'max_depth':6, 'eta':0.01, 'objective':'binary:logistic', 'subsample': 0.5, 
         'scale_pos_weight': 0.32, 
         }
param["eval_metric"] = "auc" 
#num_round = 1000000
num_round = 1 # since no improvement when continue to run xgboost
bst2 = xgb.train(param, dtrain2_lasso, num_round, evals = evallist2_lasso)
#bst2 = xgb.train(param, dtrain2_lasso, num_round, evals = evallist2_lasso, early_stopping_rounds = 10000)
#num_round = 4000
#bst = xgb.train(param, dtrain_lasso, num_round, evals = evallist_lasso)
#preds2 = bst2.predict(dtest2_lasso)
print(calc_accuracy( (preds2 > 0.5) * 1, y_test_sc2))
confusion_matrix((preds2 > 0.5) * 1, y_test_sc2)

# ----------------------------------------------------
# sub-challenge 3
# ----------------------------------------------------

#sc3 = pd.concat([cn_ge, phenotype3], axis = 1)
#encoded_sc3 = pd.get_dummies(sc3)

# fix the columns as sc1
X3_phenotype = pd.get_dummies(phenotype3.copy())
diff_cols = np.setdiff1d(dummy_phenotype1.columns.values, X3_phenotype.columns.values)
diff_cols_pd = pd.DataFrame(np.zeros((X3_phenotype.shape[0], len(diff_cols))),
                            columns = diff_cols)
diff_cols_pd.index = X3_phenotype.index # set rownames
X3_phenotype_extend = pd.concat([X3_phenotype, diff_cols_pd], axis = 1)
X3 = pd.concat([cn_ge, X3_phenotype_extend], axis = 1)[sc1.columns.values]

X_train_sc3, X_test_sc3, y_train_sc3, y_test_sc3 = train_test_split(X3, survival_outcome3, random_state = 0, test_size = 0.2)
dtrain3 = xgb.DMatrix(X_train_sc3.values[:, features_selected_from_sc1],
                      label = y_train_sc3)
dtest3 = xgb.DMatrix(X_test_sc3.values[:, features_selected_from_sc1],
                      label = y_test_sc3)

param = {'max_depth':6, 'eta':0.001, 'objective':'binary:logistic', 'subsample': 0.3, 
         #'scale_pos_weight': 0.15, 
#         'max_delta_step': 1,
 #        'gamma': 0.5
         }
#num_round = 1000000
num_round = 1000
param["eval_metric"] = "auc" 
evallist3 = [(dtrain3, 'train'), (dtest3, 'eval')]
#bst = xgb.train(param, dtrain_lasso, num_round, evals = evallist_lasso, early_stopping_rounds = 1000)
bst3 = xgb.train(param, dtrain3, num_round, evals = evallist3)

# get feature
sorted_features_by_xgb3 = [k for k, v in sorted(bst3.get_fscore().items(), key = lambda item: item[1])]
# top 30 predictors
#for fname in sorted_features_by_xgb3[::-1][0:30]:
for fname in sorted_features_by_xgb3[::-1]:
    idx = int(fname.split("f")[1])
    print(X_train_sc1.columns.values[lasso_idx[idx] - 1])


preds3 = bst3.predict(dtest3)
print(calc_accuracy( (preds3 > 0.5) * 1, y_test_sc3))
confusion_matrix((preds3 > 0.5) * 1, y_test_sc3)
fpr, tpr, thres = roc_curve(y_test_sc3, preds3)
auc(fpr, tpr) 


# ----------------------------------------------------
# Phase 2
# ----------------------------------------------------
p2_expression = pd.read_csv(f"{folder}/sc1_Phase2_GE_FeatureMatrix.tsv", header = 0, index_col = 0, sep = "\t")
p2_phenotype1 = pd.read_csv(f"{folder}/sc1_Phase2_GE_Phenotype.tsv", header = 0, index_col = 0, sep = "\t", na_values = [" ", ""])
p2_dummy_phenotype1 = pd.get_dummies(p2_phenotype1.copy()) 

# predict sc1
p2_diff_cols = np.setdiff1d(dummy_phenotype1.columns.values, p2_dummy_phenotype1.columns.values)
p2_diff_cols_pd = pd.DataFrame(np.zeros((p2_dummy_phenotype1.shape[0], len(p2_diff_cols))),
                            columns = p2_diff_cols)
p2_diff_cols_pd.index = p2_dummy_phenotype1.index
p2_dummy_phenotype1_extend = pd.concat([p2_dummy_phenotype1, p2_diff_cols_pd], axis = 1)[dummy_phenotype1.columns.values]
p2_Xtest = pd.concat([p2_expression, p2_dummy_phenotype1_extend], axis = 1)

p2_preds1 = (bst.predict(xgb.DMatrix(p2_Xtest.values[:, lasso_idx - 1])) > 0.5) * 1
header = np.array(["PATIENTID", "SURVIVAL_STATUS"])
np.savetxt("subchallenge_1.tsv", np.vstack((header, np.hstack((p2_expression.index.values.reshape(-1,1), p2_preds1.astype("int").reshape(-1, 1))))), delimiter='\t', fmt = "%s")

# predict sc2
p2_copy_number = pd.read_csv(f"{folder}/sc2_Phase2_CN_FeatureMatrix.tsv", header = 0, index_col = 0, sep = "\t")
p2_phenotype2 = pd.read_csv(f"{folder}/sc2_Phase2_CN_Phenotype.tsv", header = 0, index_col = 0, sep = "\t", na_values = [" ", ""])
p2_dummy_phenotype2 = pd.get_dummies(p2_phenotype2.copy())

p2_diff_cols2 = np.setdiff1d(dummy_phenotype2.columns.values, p2_dummy_phenotype2.columns.values)
p2_diff_cols2_pd = pd.DataFrame(np.zeros((p2_dummy_phenotype2.shape[0], len(p2_diff_cols2))), columns = p2_diff_cols2)
p2_diff_cols2_pd.index = p2_dummy_phenotype2.index
p2_dummy_phenotype2_extend = pd.concat([p2_dummy_phenotype2, p2_diff_cols2_pd], axis = 1)[dummy_phenotype2.columns.values]
p2_Xtest2 = pd.concat([p2_copy_number, p2_dummy_phenotype2_extend], axis = 1)

p2_preds2 = cvglmnetPredict(cvfit2, p2_Xtest2.values, s = "lambda_min", ptype = "class")
# or directly use the majority rule
# p2_preds2 = np.ones(p2_copy_number.shape[0])
np.savetxt("subchallenge_2.tsv", np.vstack((header, np.hstack((p2_copy_number.index.values.reshape(-1,1), p2_preds2.astype("int").reshape(-1, 1))))), delimiter='\t', fmt = "%s")

p2_cn_ge = pd.read_csv(f"{folder}/sc3_Phase2_CN_GE_FeatureMatrix.tsv", header = 0, index_col = 0, sep = "\t")
p2_phenotype3 = pd.read_csv(f"{folder}/sc3_Phase2_CN_GE_Phenotype.tsv", header = 0, index_col = 0, sep = "\t", na_values = [" ", ""])
p2_dummy_phenotype3 = pd.get_dummies(p2_phenotype3.copy()) 

p2_diff_cols3 = np.setdiff1d(dummy_phenotype1.columns.values, p2_dummy_phenotype3.columns.values)
p2_diff_cols3_pd = pd.DataFrame(np.zeros((p2_dummy_phenotype3.shape[0], len(p2_diff_cols3))),
                            columns = p2_diff_cols3)
p2_diff_cols3_pd.index = p2_dummy_phenotype3.index
p2_dummy_phenotype3_extend = pd.concat([p2_dummy_phenotype3, p2_diff_cols3_pd], axis = 1)[dummy_phenotype1.columns.values]
p2_Xtest3 = pd.concat([p2_cn_ge, p2_dummy_phenotype3_extend], axis = 1)

p2_preds3 = (bst3.predict(xgb.DMatrix(p2_Xtest3.values[:, features_selected_from_sc1])) > 0.5)* 1.0
np.savetxt("subchallenge_3.tsv", np.vstack((header,np.hstack((p2_cn_ge.index.values.reshape(-1,1), p2_preds3.astype("int").reshape(-1, 1))))), delimiter='\t', fmt = "%s")
