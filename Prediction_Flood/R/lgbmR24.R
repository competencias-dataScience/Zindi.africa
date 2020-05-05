# Note: have present that less iterations give better results

# load data
load("../newData/dataTrain.Rdata")
load("../newData/dataTest.Rdata")

# New data --> rate precipitation
dataTrain <- dataTrain %>% 
  mutate(RS1 = PS1/prec_total,
         RS2 = PS2/prec_total,
         RS3 = PS3/prec_total,
         RS4 = PS4/prec_total,
         RS5 = PS5/prec_total,
         RS6 = PS6/prec_total,
         RS7 = PS7/prec_total,
         RS8 = PS8/prec_total,
         RS9 = PS9/prec_total,
         RS10 = PS10/prec_total,
         RS11 = PS11/prec_total,
         RS12 = PS12/prec_total,
         RS13 = PS13/prec_total,
         RS14 = PS14/prec_total,
         RS15 = PS15/prec_total,
         RS16 = PS16/prec_total,
         RS17 = PS17/prec_total)

dataTest <- dataTest %>% 
  mutate(RS1 = PS1/prec_total,
         RS2 = PS2/prec_total,
         RS3 = PS3/prec_total,
         RS4 = PS4/prec_total,
         RS5 = PS5/prec_total,
         RS6 = PS6/prec_total,
         RS7 = PS7/prec_total,
         RS8 = PS8/prec_total,
         RS9 = PS9/prec_total,
         RS10 = PS10/prec_total,
         RS11 = PS11/prec_total,
         RS12 = PS12/prec_total,
         RS13 = PS13/prec_total,
         RS14 = PS14/prec_total,
         RS15 = PS15/prec_total,
         RS16 = PS16/prec_total,
         RS17 = PS17/prec_total)

# Selection of variables for analysis 
library(tidyverse)
dataTrain <- dataTrain %>% select(long:LC_Type1, rugos_near8,
                                  RS1:RS17)

# caret for partition data
library(caret)
set.seed(123)
indx <- createDataPartition(y = dataTrain$target_2015, times = 1, p = 0.80, list = FALSE)
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
  learning_rate = 0.01,
  feature_fraction = 1,
  bagging_fraction = 1,
  min_data_in_leaf = 100,
  max_depth = -1
  #  num_leaves = 80
)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 30000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 500)

# Predictions 2019
predicciones <- predict(modelo, data.matrix(dataTest %>%
                                              select(long:LC_Type1, rugos_near8,
                                                     RS1:RS17)),
                        num_iteration = modelo$best_iter)
x11();hist(predicciones)
predicciones[predicciones > 1] <- 1

# Submission
dataTest %>% 
  select(Square_ID) %>% 
  mutate(target = predicciones) ->
  lgbmR24

# Export submission for zindi
write.csv(lgbmR24, file = "Submission/lgbmR24.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE)
x11()
impModelo %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()