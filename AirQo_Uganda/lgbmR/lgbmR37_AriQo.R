# Load data
load("../my_data/dataTrain_Originales2.Rdata")
load("../my_data//dataTest_Originales2.Rdata")

# Selection of variables for analysis 
library(tidyverse)
dataTrain <- dataTrain2 %>% select(-c(ID)) %>% 
  select(target, everything())

# Categorical features
catfeatures <- names(dataTrain %>% select_if(is.factor))

# caret for partition data
library(caret)
set.seed(123)
myFolds <- createFolds(y = dataTrain$target, k = 15, list = TRUE)
index <-list()
index$train <- lapply(myFolds, function(x) which(!1:nrow(dataTrain) %in% x))
index$test<-lapply(myFolds, function(x) which(1:nrow(dataTrain) %in% x))

# Parameters for lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "regression",
  metric = "rmse",
  learning_rate = 0.01,
  feature_fraction = 0.3,
  bagging_fraction = 1,
  min_data_in_leaf = 100,
  num_leaves = 255,
  max_depth = -1
)

# lightgbm with K-Fold (k = 10)
library(lightgbm)
k <- 15
predTest <- list()
bestScore <- c()
for (i in 1:k) {
  
  # Data train and test
  dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dataTrain[index$train[[i]], -1]),
                                label = dataTrain[index$train[[i]], 1],
                                categorical_feature = catfeatures)
  dataTest_lgbm <- lgb.Dataset(data = data.matrix(dataTrain[index$test[[i]], -1]),
                               label = dataTrain[index$test[[i]], 1],
                               categorical_feature = catfeatures)
  
  # Train model
  model <- lgb.train(params = myParams,
                     data = dataTrain_lgbm,
                     nrounds = 30000,
                     valids = list(test = dataTest_lgbm),
                     early_stopping_rounds = 500)
  
  # Predictions
  predictions <- predict(model,
                         data.matrix(dataTest2 %>% select(-ID)),
                         num_iteration = model$best_iter)
  predTest[[i]] = predictions
  
  # Best score for iteration
  bestScore[i] = model$best_score
  
  # Next iteration
  cat("Iteration:==========", i, "RSME:==========", model$best_score, "Ready!")
}

# Mean predictions
dataPred <- as.data.frame(predTest)
names(dataPred) <- paste0("Mod", 1:15)
predicciones <- apply(dataPred, 1, mean)

#predicciones[predicciones < 0] <- 0
x11();hist(predicciones)

# Submission 
dataTest2 %>% 
  select(ID) %>% 
  mutate(target = predicciones) ->
  lgbmR37_AirQo

# Export submission for zindi
write.csv(lgbmR37_AirQo, file = "Submission/lgbmR37_AirQo.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE,)
x11()
impModelo %>% 
  slice(1:50) %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()