# Importando bibliotecas
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

# Datos train
nombres_train = pd.read_csv('../my_data/nombres_train_nuevos.csv', header=None, sep=',')

df_train = pd.read_csv('../my_data/dataTrainEdimer_pyLog.csv', names = nombres_train[1:][0],
                       header=None, sep=',', encoding = "latin",)[1:]
df_train.head(10)

# Datos test
nombres_test = pd.read_csv('../my_data/nombres_test_nuevos.csv', header=None, sep=',')
df_test = pd.read_csv('../my_data/dataTestEdimer_pyLog.csv', names = nombres_test[1:][0],
                       header=None, sep=',', encoding = "latin",)[1:]
df_test.head(10) 

# Coerción a numérico
cols = df_train.columns.drop(["ID"])
df_train[cols] = df_train[cols].apply(pd.to_numeric, errors='coerce')

colsT = df_test.columns.drop(["ID"])
df_test[colsT] = df_test[colsT].apply(pd.to_numeric, errors='coerce')

# Split data
df_train2, df_test2 = train_test_split(df_train, test_size = 0.2, random_state = 123)

# Datos para entrenamiento del modelo
y_train = np.exp(df_train2["target"])
y_test = np.exp(df_test2["target"])
X_train = df_train2.drop(["ID", "target"], axis=1)
X_test = df_test2.drop(["ID", "target"], axis=1)

# Dataset lgb, igual que para XGBoost
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Parámetros del modelo
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'rmse'},
    'num_leaves': 255,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'verbose': 1,
    'max_depth': -1,
    'min_data_in_leaf': 100
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50,
                categorical_feature=["location"])

# Predicciones Test (submission)
print('Starting predict...')
predichos = gbm.predict(data=df_test.drop(["ID"], axis = 1))
predichos


# Exportando predicciones
mi_array = {'ID': df_test['ID'],
            'target': predichos}
          
s8_lgb = pd.DataFrame(data = mi_array)
s8_lgb.to_csv('submissions/s8_lgb.csv', index = False, header=True)



# =================================== Otro modelo S9 python (RMSE Valid: 0.25...)

# A la variable respuesta le aplico np.exp()--> original y NO logaritmo

# Parámetros del modelo
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'rmse'},
    'num_leaves': 127,
    'learning_rate': 0.1,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'verbose': 1,
    'max_depth': 8,
    'min_data_in_leaf': 200
    'lambda_l1': 5
    'min_gain_to_split': 1
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50,
                categorical_feature=["location"])

# Predicciones Test (submission)
print('Starting predict...')
predichos = gbm.predict(data=df_test.drop(["ID"], axis = 1))
predichos


# Exportando predicciones
mi_array = {'ID': df_test['ID'],
            'target': predichos}
          
s9_lgb = pd.DataFrame(data = mi_array)
s9_lgb.to_csv('submissions/s9_lgb.csv', index = False, header=True)


# =================================== Otro modelo S10 python (RMSE Valid: 25.0803)

# A la variable respuesta le aplico np.exp()--> original y NO logaritmo

# Parámetros del modelo
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'rmse'},
    'num_leaves':500,
    'learning_rate': 0.1,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'verbose': 1,
    'max_depth': -1,
    'min_data_in_leaf': 200
    'lambda_l1': 1
    'min_gain_to_split': 5
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50,
                categorical_feature=["location"])

# Predicciones Test (submission)
print('Starting predict...')
predichos = gbm.predict(data=df_test.drop(["ID"], axis = 1))
predichos


# Exportando predicciones
mi_array = {'ID': df_test['ID'],
            'target': predichos}
          
s10_lgb = pd.DataFrame(data = mi_array)
s10_lgb.to_csv('submissions/s10_lgb.csv', index = False, header=True)


# =================================== Otro modelo S11 python (RMSE Valid: )

# Variable respuesta como logaritmo y cambio la semilla a 73

# Split data
df_train2, df_test2 = train_test_split(df_train, test_size = 0.15, random_state = 73)

# Datos para entrenamiento del modelo
y_train = df_train2["target"]
y_test = df_test2["target"]
X_train = df_train2.drop(["ID", "target"], axis=1)
X_test = df_test2.drop(["ID", "target"], axis=1)

# Dataset lgb, igual que para XGBoost
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Parámetros del modelo
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'rmse'},
    'num_leaves':300,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'verbose': 1,
    'max_depth': -1,
    'min_data_in_leaf': 200
    'lambda_l1': 1
    'min_gain_to_split': 5
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50,
                categorical_feature=["location"])

# Predicciones Test (submission)
print('Starting predict...')
predichos = gbm.predict(data=df_test.drop(["ID"], axis = 1), num_iteration=gbm.best_iteration)
predichos = np.exp(predichos)
predichos


# Exportando predicciones
mi_array = {'ID': df_test['ID'],
            'target': predichos}
          
s11_lgb = pd.DataFrame(data = mi_array)
s11_lgb.to_csv('submissions/s11_lgb.csv', index = False, header=True)
