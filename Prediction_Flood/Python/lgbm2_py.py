# Importando bibliotecas
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import KFold, train_test_split

# Datos train
nombres_train = pd.read_csv('../../newData/nombresTrain.csv', header=None, sep=',')

df_train = pd.read_csv('../../newData/dataTrainPy.csv', names = nombres_train[1:][0],
                       header=None, sep=',', encoding = "latin",)[1:]
df_train.head(10)

# Datos test
nombres_test = pd.read_csv('../../newData/nombresTest.csv', header=None, sep=',')
df_test = pd.read_csv('../../newData/dataTestPy.csv', names = nombres_test[1:][0],
                      header=None, sep=',', encoding = "latin",)[1:]
df_test.head(10) 

# Coerción a numérico
cols = df_train.columns.drop(["Square_ID"])
df_train[cols] = df_train[cols].apply(pd.to_numeric, errors='coerce')

colsT = df_test.columns.drop(["Square_ID"])
df_test[colsT] = df_test[colsT].apply(pd.to_numeric, errors='coerce')

# Split data
df_train2, df_test2 = train_test_split(df_train, test_size = 0.2, random_state = 123)

# Datos para entrenamiento del modelo
y_train = df_train2["target_2015"]
y_test = df_test2["target_2015"]
X_train = df_train2.drop(["Square_ID", "target_2015"], axis=1)
X_test = df_test2.drop(["Square_ID", "target_2015"], axis=1)

# Dataset lgb, igual que para XGBoost
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data = True)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data = True)

# Parámetros del modelo
params = {
  'boosting': 'gbdt',
  'objective': 'poisson',
  'metric': {'rmse'},
  'learning_rate': 0.05,
  'feature_fraction': 0.4,
  'bagging_fraction': 0.8,
  'bagging_freq': 1,
  'max_depth': -1,
  'min_data_in_leaf': 15,
  'num_iterations': 7500,
  'max_leaves': 500
}

# Modelo 
gbm = lgb.train(params,
                lgb_train,
                verbose_eval=500,
                valid_sets=[lgb_eval],
                early_stopping_rounds=100)

# Predicciones en train
pred_train = gbm.predict(X_train)

# RMSE train
np.sqrt(mean_squared_error(y_train, pred_train))


# Validación cruzada para predicciones
errlgb = []
y_pred_totlgb = []
Xtest = df_test.drop(["Square_ID"], axis = 1)
dfTrain = df_train2.drop(["Square_ID", "target_2015"], axis=1)
dfTest = df_train2['target_2015'].values

fold = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in fold.split(dfTrain):
    
    X_train, X_test = dfTrain.loc[train_index], dfTrain.loc[test_index]
    y_train, y_test = dfTest[train_index], dfTest[test_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    
    clf = lgb.train(params=params, 
                     early_stopping_rounds=200,
                     verbose_eval=500,
                     train_set=lgb_train,
                     valid_sets=[lgb_eval])

    y_pred = clf.predict(X_test) 

    print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    
    errlgb.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    p = clf.predict(Xtest)
    
    y_pred_totlgb.append(p)

# Predicciones promedio
predichos = np.mean(y_pred_totlgb,0)

# Predicciones Test (submission)
print('Starting predict...')
#predichos = gbm.predict(data=df_test.drop(["Square_ID"], axis = 1))
predichos[predichos < 0] = 0
predichos[predichos > 1] = 1
predichos


# Exportando predicciones
mi_array = {'Square_ID': df_test['Square_ID'],
  'target': predichos}

s2_lgb = pd.DataFrame(data = mi_array)
s2_lgb.to_csv('Submission_Py/s2_lgb.csv', index = False, header=True)
