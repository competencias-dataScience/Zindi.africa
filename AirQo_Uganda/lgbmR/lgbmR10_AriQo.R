# Load data
load("../my_data/dataTrain_Originales.Rdata")
load("../my_data//dataTest_Originales.Rdata")

# Selection of variables for analysis 
library(tidyverse)
dataTrain <- dataTrain %>% select(-c(ID)) %>% 
  select(target, everything())

# caret for partition data
library(caret)
set.seed(123)
indx <- createDataPartition(y = dataTrain$target, times = 1, p = 0.70,
                            list = FALSE)
dfTrain <- dataTrain[indx, ]
dfTest <- dataTrain[-indx, ]

# Data for lightgbm (all less colum 3-->target)
library(lightgbm)
dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dfTrain[, -1]),
                              label = dfTrain[, 1],
                              categorical_feature = "location")
dataTest_lgbm <- lgb.Dataset(data = data.matrix(dfTest[, -1]),
                             label = dfTest[, 1],
                             categorical_feature = "location")

# Parameters for lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "regression",
  metric = "rmse",
  learning_rate = 0.05,
  feature_fraction = 0.9,
  bagging_fraction = 0.9,
  min_data_in_leaf = 100,
  num_leaves = 255,
  max_depth = -1
)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 10000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 500)

# Predictions
predicciones <- predict(modelo, data.matrix(dataTest %>% select(-ID)), 
                        num_iteration = modelo$best_iter)
predicciones[predicciones < 0] <- 0
x11();hist(predicciones)

# Submission lgbmR9_AirQo
dataTest %>% 
  select(ID) %>% 
  mutate(target = predicciones) ->
  lgbmR10_AirQo

# Export submission for zindi
write.csv(lgbmR10_AirQo, file = "Submission/lgbmR10_AirQo.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE)
x11()
impModelo %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()