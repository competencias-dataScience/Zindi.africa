# Import data
load("myData/Train1.Rdata")
load("myData/Test1.Rdata")
dataSampleSubm <- read.csv("data/updated_ss.csv")

# Partition data
library(caret)
set.seed(123)
indx <- createDataPartition(y = dfTrain$myTarget, times = 1, p = 0.7, list = FALSE)
trainData <- dfTrain[indx, ]
validData <- dfTrain[-indx, ]

# Data for lightgbm (all less colum 3-->target)
library(lightgbm)
dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dfTrain[, -1]), label = dfTrain[, 1])
dataTest_lgbm <- lgb.Dataset(data = data.matrix(dfTest[, -1]), label = dfTest[, 1])

# Parameters for lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "binary",
  metric = "binary_logloss",
  learning_rate = 0.05,
  feature_fraction = 1,
  bagging_fraction = 1,
  max_depth = -1,
  is_unbalance = TRUE)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 30000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 500)

# Predictions
predicciones <- predict(modelo, data.matrix(dfTest))

# Submission lgbmR1
library(dplyr)
dataSampleSubm %>% 
  select(ID) %>% 
  mutate(target = predicciones) ->
  lgbmR1

# Export submission for zindi
write.csv(lgbmR1, file = "Submission/lgbmR1.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE)
x11()
impModelo %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()
