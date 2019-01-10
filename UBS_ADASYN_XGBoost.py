##UBSSynthetic Minority Oversampling Technique (SMOTE) XGBoost
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset0 = pd.read_csv('UBS_branches_data.csv')
dataset1=dataset0.drop_duplicates(subset=None, keep='first', inplace=False)##data cleaning, remove the repeat rows
dataset=dataset1[ dataset1['CBSAPopulation'] > 0]##data cleaning, only keep the zipcodes whose polulation are not 0
X = dataset.iloc[:,1:29].values ##features
X_id=dataset.iloc[:, 0].values
y0 = dataset.iloc[:,31].values##lables: indicate whether the zipcode contains branch or not
y = dataset.iloc[:,32].values

for yi in range(0,len(y0)):
    if y0[yi]>=1:
       y[yi]=1 ### if the zipcodes contains branch/braches lable it as 1, otherwise lable it as 0


####ADASYN sampling
from imblearn.over_sampling import ADASYN
oversampler=ADASYN(random_state=0)
X_extend,y_extend=oversampler.fit_sample(X,y)

####Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_extend_train, X_extend_test, y_extend_train, y_extend_test = train_test_split(X_extend, y_extend, test_size = 0.2, random_state = 0)

##XGboost libraries
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

##Evaluate trained XGBRegressor model 
#######XGBboost classifier 
## Step 1 : Fix learning rate and number of estimators for tuning tree-based parameters
#xgb1 = XGBClassifier(
#    learning_rate =0.019,
#    n_estimators=1200,#530,
#    max_depth=9,
#    min_child_weight=1, 
#    gamma=0.1,#0.3,#0.3,0.2,
#    subsample=0.8,
#    colsample_bytree=0.4,#0.8,#0.2,#0.8,#0.4,
#    objective= 'binary:logistic', 
#    nthread=4,
#    scale_pos_weight=1,
#    seed=27
#    )
###  modelfit(xgb1, X_train, y_train)
#xgb1.fit(X_extend_train, y_extend_train)
#
##  Performance of train set
#auc_train = roc_auc_score(y_extend_train, xgb1.predict(X_extend_train))
#print("Performance of train set : ",auc_train)
#
## Performance of test set
#auc_test = roc_auc_score(y_extend_test, xgb1.predict(X_extend_test))
#print("Performance of test set : ", auc_test)



##Model training with different hyper parameters based on cross validation and gridsearch
#######XGBboost classifier tuning
##Tune n_estimators
#param_test2 = {
#    'n_estimators':range(100,2000,100)#(1,6,2)
#}
#gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.019, 
#                                                  n_estimators=1000, 
#                                                  max_depth=9,
#                                                  min_child_weight=1,
#                                                  gamma=0.3, 
#                                                  subsample=0.8, 
#                                                  colsample_bytree=0.8,
#                                                  objective= 'binary:logistic', 
#                                                  nthread=4, 
#                                                  scale_pos_weight=1,
#                                                  seed=27), 
#
#                                                  param_grid = param_test2, 
#                                                  scoring='roc_auc',
#                                                  verbose=10,
#                                                  n_jobs=1,##only one CPU
#                                                  iid=False, 
#                                                  cv=5)
#
#gsearch2.fit(X_extend,y_extend)
#gsearch2.best_params_, gsearch2.best_score_


##Tune max_depth and min_child_weight
#param_test1 = {
#    'max_depth':range(3,15,2),#(3,10,2),
#    'min_child_weight':range(1,11,2)#(1,6,2)
#}
#gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, 
#                                                  n_estimators=403,#1000, 
#                                                  max_depth=9,
#                                                  min_child_weight=1,
#                                                  gamma=0.2, 
#                                                  subsample=0.8, 
#                                                  colsample_bytree=0.4,
#                                                  objective= 'binary:logistic', 
#                                                  nthread=4, 
#                                                  scale_pos_weight=1,
#                                                  seed=27), 
#
#                                                  param_grid = param_test1, 
#                                                  scoring='roc_auc',
#                                                  verbose=10,
#                                                  n_jobs=1,##only one CPU
#                                                  iid=False, 
#                                                  cv=5)
#
#gsearch1.fit(X_extend,y_extend))
#gsearch1.best_params_, gsearch1.best_score_



####Tune gamma
#param_test3 = { 
#    'gamma':[i/10.0 for i in range(0,5)] 
#}
#gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.019, 
#                                                  n_estimators=1200, 
#                                                  max_depth = 9,
#                                                  min_child_weight = 1,
#                                                  gamma=0.3, subsample=0.8, colsample_bytree=0.8,
#                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#                        param_grid = param_test3, 
#                        scoring='roc_auc',
#                         verbose=10,
#                       n_jobs=1,##only one CPU
#                        iid=False, 
#                        cv=5)
#
#gsearch3.fit(X_extend,y_extend)
##gsearch3.grid_scores_, 
#gsearch3.best_params_, gsearch3.best_score_



##Tune subsample and colsample_bytree
#param_test4 = { 
##     'subsample':[i/100.0 for i in range(55,70,5)],
## 'colsample_bytree':[i/100.0 for i in range(60,80,5)]
# # 'subsample':[i/10.0 for i in range(2,10,2)],
#    'colsample_bytree':[i/10.0 for i in range(2,10,2)],
#    'gamma':[i/10.0 for i in range(0,5)] 
#}
#gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.019, 
#                                                  n_estimators=1200,#403, 
#                                                  max_depth = 9,
#                                                  min_child_weight = 1,
#                                                  gamma=0.3, subsample=0.8, colsample_bytree=0.2,#0.4,
#                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#                        param_grid = param_test4, 
#                        scoring='roc_auc',
#                         verbose=10,
#                       n_jobs=1,##only one CPU
#                        iid=False, 
#                        cv=5)
#
#gsearch4.fit(X_extend,y_extend)
##gsearch4.grid_scores_, 
#gsearch4.best_params_, gsearch4.best_score_



####tunning scale_pos_weight
#param_test5 = {
# 'scale_pos_weight':[i for i in range(1,10,1)]
#}
#gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.005, 
#                                                  n_estimators=403, 
#                                                  gamma=0.3,
#                                                  subsample=0.8,
#                                                  colsample_bytree=0.2,#0.8,
#                                                  max_depth = 9,
#                                                  min_child_weight = 1,
#                                                  objective= 'binary:logistic',scale_pos_weight=1, nthread=4, seed=27), 
#                        param_grid = param_test5, 
#                        scoring='roc_auc',
#                          verbose=10,
#                        n_jobs=1,
#                        iid=False, 
#                        cv=5)
#
#gsearch5.fit(X_extend,y_extend))
##gsearch6.grid_scores_, 
#gsearch5.best_params_, gsearch5.best_score_



##Reducing Learning Rate in order to reduce overfit
#param_test6 = {
# 'learning_rate':[i/1000.0 for i in range(5,20,2)]
#}
#gsearch6 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.019, 
#                                                  n_estimators=403, 
#                                                  gamma=0.3,
#                                                  subsample=0.8,
#                                                  colsample_bytree=0.2,#0.8,
#                                                  max_depth = 9,
#                                                  min_child_weight = 1,
#                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#                        param_grid = param_test6, 
#                        scoring='roc_auc',
#                          verbose=10,
#                        n_jobs=1,
#                        iid=False, 
#                        cv=5)
#
#gsearch6.fit(X_extend,y_extend))
##gsearch6.grid_scores_, 
#gsearch6.best_params_, gsearch6.best_score_





###Model trained, and optimal hyper parameters are obtained
### Fix new param with optimal parameter
xgb6 = XGBClassifier(
    learning_rate =0.019, n_estimators=1200,#530,#403,#1800,
    gamma=0.1,#0.2,#0.3,#0.0,
    subsample=0.8,
    colsample_bytree=0.4,#0.8,#0.2,#0.8,#0.4,
    objective= 'binary:logistic', nthread=4,scale_pos_weight=1,
    seed=27,
    max_depth = 9,
    min_child_weight = 1
)

xgb6.fit(X_extend_train,y_extend_train)




###plot forest tree
from xgboost import plot_tree
plot_tree(xgb6)
plt.show()
################
## to see the importance ranking of the each feature
import xgboost as xgb
from xgboost import plot_importance
params = {
    #'booster': 'gbtree',
   # 'learning_rate'=0.005
    'scale_pos_weight': 1,
   # 'n_estimators'=100,
    'objective': 'binary:logistic',
    'gamma':0.1,#0.2,#0.3,#0.0,
    'max_depth': 9,
     'min_child_weight': 1,
    #'lambda': 3,
    'subsample': 0.8,
    'colsample_bytree':0.4,# 0.8,#0.2,#0.8,#0.4,
   
    'silent': 1,
    'seed': 27,
   'nthread': 4,
}

dtrain = xgb.DMatrix(X_extend_train, label=y_extend_train)
dtest = xgb.DMatrix(X_extend_test, label=y_extend_test)


num_rounds = 1000
plst = params.items()
watchlist = [ (dtrain,'train'), (dtest, 'test') ]
xgb_model = xgb.train(params, dtrain, num_rounds, watchlist )


# show important features
plot_importance(xgb_model )
plt.show()
#########################model evalutation
from sklearn.metrics import confusion_matrix
y_pred=xgb6.predict(X)
y_pred_prob=xgb6.predict_proba(X)
cm=confusion_matrix(y,y_pred)


tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
# Sensitivity : 
sens_ = tp / (tp + fn)
print("Sensitivity/Recall  on test set : ", sens_)
# Specificity 
sp_ = tn / (tn + fp)
print("Specificity  on test set : ", sp_)
# False positive rate (FPR)
FPR = fp / (tn + fp)
print("False positive rate  on test set : ", FPR)

# Error rate : 
err_rate = (fp + fn) / (tp + tn + fn + fp)
print("Error rate  on test set : ", err_rate)
# Accuracy : 
acc_ = (tp + tn) / (tp + tn + fn + fp)
print("Accuracy on test set  : ", acc_)


# ROC CURVE
y_probas  = xgb_model.predict(dtest)
from sklearn.metrics import roc_curve, auc 
fpr, tpr, thresholds = roc_curve(y_extend_test, y_probas)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#######weighting  and ranking for predicted probability
w1=10/30398
w0=259/30398

######weight for predicting class ``1"
weighted_y_pred_prob1=[0]*len(y_pred)
y_pred_prob1=y_pred_prob[:,1]
for i in range(0,len(y_pred)-1):
    if y_pred[i]==0:
       weighted_y_pred_prob1[i]=w0*y_pred_prob1[i]
    elif y_pred[i]==1:
       weighted_y_pred_prob1[i]=w1*y_pred_prob1[i]
    

##################ranking for weighted predicted probability
df_wypp1=pd.DataFrame(weighted_y_pred_prob1)
dscnd_wypp1=df_wypp1.sort_values(by=0,ascending=False)
dscnd_wypp1_id=dscnd_wypp1.iloc[:, 0].index

##most likely 10 ID 
top10_class1_idex=dscnd_wypp1_id[0:10]
Final_top10_class1_prob_id=X_id[top10_class1_idex]
print('The most likely 10 zipcodes to be next new braches are:', Final_top10_class1_prob_id)


