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
  objective = "regression_l2",
  metric = "rmse",
  learning_rate = 0.01,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  min_data_in_leaf = 300
)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 5000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 200)

# Predictions
predicciones <- predict(modelo, data.matrix(dataTest %>% select(-ID)))
predicciones[predicciones < 0] <- 0
x11();hist(predicciones)

# Submission lgbmR7_AirQo
dataTest %>% 
  select(ID) %>% 
  mutate(target = predicciones) ->
  lgbmR7_AirQo

# Export submission for zindi
write.csv(lgbmR7_AirQo, file = "Submission/lgbmR7_AirQo.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE)
x11()
impModelo %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()