# Importando bibliotecas
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import KFold, train_test_split

# Datos train
nombres_train = pd.read_csv('../../newData/nombresTrain2.csv', header=None, sep=',')

df_train = pd.read_csv('../../newData/dataTrainPy2.csv', names = nombres_train[1:][0],
                       header=None, sep=',', encoding = "latin",)[1:]
df_train.head(10)

# Datos test
nombres_test = pd.read_csv('../../newData/nombresTest2.csv', header=None, sep=',')
df_test = pd.read_csv('../../newData/dataTestPy2.csv', names = nombres_test[1:][0],
                      header=None, sep=',', encoding = "latin",)[1:]
df_test.head(10) 

# Coerción a numérico
cols = df_train.columns.drop(["Square_ID"])
df_train[cols] = df_train[cols].apply(pd.to_numeric, errors='coerce')

colsT = df_test.columns.drop(["Square_ID"])
df_test[colsT] = df_test[colsT].apply(pd.to_numeric, errors='coerce')

# Parámetros del modelo (todos por default, excepto el número de iteraciones que es 100 por default)
params = {
  'boosting': 'gbdt',
  'objective': 'poisson',
  'metric': 'rmse',
  'learning_rate': 0.05,
  'feature_fraction': 0.5,
  'bagging_fraction': 0.8,
  #  'max_depth': -1,
  'min_data_in_leaf': 300, 
  #  'num_iterations': 500,
  #  'num_leaves': 500, 
  #  'max_bin': 2500,
}


# Modelo  con validación cruzada para predicciones
errlgb = []
y_pred_totlgb = []
Xtest = df_test.drop(["Square_ID"], axis = 1)
dfTrain = df_train.drop(["Square_ID", "target_2015"], axis = 1)
dfTest = df_train['target_2015'].values

#grupos = np.quantile(df_train["target_2015"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
fold = KFold(n_splits=10, shuffle = True, random_state = 1992)

for train_index, test_index in fold.split(dfTrain):
  
  X_train, X_test = dfTrain.loc[train_index], dfTrain.loc[test_index]
y_train, y_test = dfTest[train_index], dfTest[test_index]

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

clf = lgb.train(params = params, 
                early_stopping_rounds = 500,
                verbose_eval = 200,
                train_set = train_data,
                valid_sets = test_data)

y_pred = clf.predict(X_test) 

print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

errlgb.append(np.sqrt(mean_squared_error(y_test, y_pred)))

p = clf.predict(Xtest)

y_pred_totlgb.append(p)

# Predicciones promedio
predichos = np.mean(y_pred_totlgb,0)

# Predicciones Test (submission)
predichos[predichos < 0] = 0
predichos[predichos > 1] = 1
predichos

# Exportando predicciones
mi_array = {'Square_ID': df_test['Square_ID'],
  'target': predichos}

s10_lgb = pd.DataFrame(data = mi_array)
s10_lgb.to_csv('Submission_Py/s10_lgb.csv', index = False, header=True)
