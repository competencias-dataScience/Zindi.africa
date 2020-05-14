# load data
load("../newData/dataTrainFeatureEng2.Rdata")
load("../newData/dataTestFeatureEng2.Rdata")

# Selection of variables for analysis 
library(tidyverse)
dataTrain <- dataTrain %>% 
  select(long:LC_Type1, rugos_near8, RS1:RS17, wcs)

# Categorical features
catfeatures <- names(dataTrain %>% select_if(is.factor))

# caret for partition data with resample
library(caret)
set.seed(123)
index <- createDataPartition(y = dataTrain$target_2015, times = 10, p = 0.70, list = TRUE)

# Parameters for lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "poisson",
  metric = "rmse",
  learning_rate = 0.005,
  feature_fraction = 1,
  bagging_fraction = 1,
  min_data_in_leaf = 100,
  max_depth = -1
)

# lightgbm with K-Fold manually (k = 10)
library(lightgbm)
k <- 10
predTest <- list()
bestScore <- c()
for (i in 1:k) {
  
  # Data train and test
  dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dataTrain[index[[i]], -3]),
                                label = dataTrain[index[[i]], 3],
                                categorical_feature = catfeatures)
  dataTest_lgbm <- lgb.Dataset(data = data.matrix(dataTrain[-index[[i]], -3]),
                               label = dataTrain[-index[[i]], 3],
                               categorical_feature = catfeatures)
  
  # Train model
  model <- lgb.train(params = myParams,
                     data = dataTrain_lgbm,
                     nrounds = 30000,
                     valids = list(test = dataTest_lgbm),
                     early_stopping_rounds = 500)
  
  # Predictions
  predictions <- predict(model,
                         data.matrix(dataTest %>%
                                       select(long:LC_Type1, rugos_near8,
                                              RS1:RS17, wcs)),
                         num_iteration = model$best_iter)
  predTest[[i]] = predictions
  
  # Best score for iteration
  bestScore[i] = model$best_score
  
  # Next iteration
  cat("Iteration:==========", i, "RSME:==========", model$best_score, "Ready!")
}

# Mean predictions
dataPred <- as.data.frame(predTest)
names(dataPred) <- paste0("Mod", 1:10)
meanPred <- apply(dataPred, 1, mean)

# Submission 92
dataTest %>% 
  select(Square_ID) %>% 
  mutate(target = meanPred) ->
  lgbmR92

# Export submission for zindi
write.csv(lgbmR92, file = "Submission/lgbmR92.csv", row.names = FALSE)

# Submission 93
pred93 <- dataPred$Mod1
pred93[pred93 > 1] <- 1
dataTest %>% 
  select(Square_ID) %>% 
  mutate(target = pred93) ->
  lgbmR93

# Export submission for zindi
write.csv(lgbmR93, file = "Submission/lgbmR93.csv", row.names = FALSE)
