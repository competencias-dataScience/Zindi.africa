# Note: have present that less iterations give better results

# load data
load("../newData/dataTrain.Rdata")
load("../newData/dataTest.Rdata")

# Selection of variables for analysis 
library(tidyverse)
dataTrain <- dataTrain %>% select(-Square_ID)

# caret for partition data
library(caret)
set.seed(123)
indx <- createDataPartition(y = dataTrain$target_2015, times = 1, p = 0.60, list = FALSE)
dfTrain <- dataTrain[indx, ]
dfTest <- dataTrain[-indx, ]

# Categorical features
catfeatures <- names(dfTrain %>% select_if(is.factor))

# Data for lightgbm (all less colum 3-->target)
library(lightgbm)
dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dfTrain[, -3]), label = dfTrain[, 3],
                              categorical_feature = catfeatures)
dataTest_lgbm <- lgb.Dataset(data = data.matrix(dfTest[, -3]), label = dfTest[, 3],
                             categorical_feature = catfeatures)

# Parameters for lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "poisson",
  metric = "rmse",
  learning_rate = 0.04,
  feature_fraction = 0.5,
  bagging_fraction = 0.8,
  min_data_in_leaf = 100,
  early_stopping_rounds = 50,
  max_depth = 10,
  num_leaves = 80
)

# Train model
modelo <- lgb.train(params = myParams, data = dataTrain_lgbm, nrounds = 1000,
                    valids = list(test = dataTest_lgbm))

# Predictions 2019
predicciones <- predict(modelo, data.matrix(dataTest %>% select(-Square_ID)),
                        num_iteration = modelo$best_iter)
x11();hist(predicciones)
#predicciones[predicciones > 1] <- 1

# Submission lgbmR7
dataTest %>% 
  select(Square_ID) %>% 
  mutate(target = predicciones) ->
  lgbmR7

# Export submission for zindi
write.csv(lgbmR7, file = "Submission/lgbmR7.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE)
x11()
impModelo %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()
