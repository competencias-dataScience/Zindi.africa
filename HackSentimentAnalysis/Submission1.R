# Import data
dataTrain <- read.csv("data/updated_train.csv")
dataTest <- read.csv("data/updated_test.csv")
dataSampleSubm <- read.csv("data/updated_ss.csv")

# Tokenization text manual
#library(tm)
# <- VCorpus(VectorSource(dataTrain$text))
#textCorpus <- tm_map(x = textCorpus, FUN = content_transformer(tolower))
#textCorpus <- tm_map(x = textCorpus, FUN = content_transformer(removePunctuation))
#textCorpus <- tm_map(x = textCorpus, FUN = content_transformer(removeNumbers))
#textCorpus <- tm_map(x = textCorpus, FUN = content_transformer(stripWhitespace))
#textCorpus <- tm_map(x = textCorpus, FUN = removeWords, stopwords("english"))
#textCorpus <- tm_map(x = textCorpus, FUN = stemDocument, language = "english")
#bagWordMatrix <- DocumentTermMatrix(x = textCorpus)

# Function for tokenization of text
source("cleanText.R")
textTrain <- cleanText(text = dataTrain$text, language = "english", bagWord = TRUE)
textTest <- cleanText(text = dataTest$text, language = "english", bagWord = TRUE)

# Data for train model classification
matrixTrain <- textTrain$bagWordMatrix
matrixTrain <- removeSparseTerms(x = matrixTrain, sparse = 0.999) 
matrixTest <- textTest$bagWordMatrix

# Corpus word
corpusTrain <- textTrain$textCorpus
#x11();wordcloud::wordcloud(corpusTrain)
corpusTest <- textTest$textCorpus

# Dataframe for modelling
dfTrain <- as.data.frame(as.matrix(matrixTrain))
dfTest <- as.data.frame(as.matrix(matrixTest))

# Adding target
library(tidyverse)
dfTrain <- dfTrain %>% 
  select(which(names(dfTrain) %in% names(dfTest))) %>% 
  mutate(myTarget = dataTrain$target) %>% 
  select(myTarget, everything()) 

dfTest <- dfTest %>% 
  select(which(names(dfTest) %in% names(dfTrain)))

# Create data partition with caret for model
library(caret)
set.seed(123)
indx <- createDataPartition(y = dfTrain$myTarget, times = 1, p = 0.8, list = FALSE)
trainData <- dfTrain[indx, ]
validData <- dfTrain[-indx, ]

# Model SVM
library(e1071)
svmModel <- svm(x = trainData[, -1],
                y = trainData$target,
                kernel = "radial",
                cost = 1,
                type = "C-classification",
                probability = TRUE)
svmModel

# Predictions train data
predictTrain <- predict(object = svmModel, newdata = trainData[, -1], probability = TRUE)
confusionMatrix(data = predictTrain, reference = factor(trainData$myTarget))
probTrain <- as.data.frame(attr(predictTrain, "probabilities"))

# Predictions validation data
predictValid <- predict(object = svmModel, newdata = validData[, -1], probability = TRUE)
confusionMatrix(data = predictValid, reference = factor(validData$myTarget))
probValid <- as.data.frame(attr(predictValid, "probabilities"))

# Predictions test data
predictTest <- predict(object = svmModel,
                       newdata = dfTest,
                       probability = TRUE)
probTest <- as.data.frame(attr(predictTest, "probabilities"))

# Submission
dataSampleSubm <- dataSampleSubm %>% 
  select(ID) %>% 
  cbind(probTest)
write.csv(x = dataSampleSubm, file = "Submission/Subm1.csv", row.names = FALSE)