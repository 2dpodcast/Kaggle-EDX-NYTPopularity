# KAGGLE COMPETITION: PREDICTING POPULARITY OF NYT-BLOG-ARTICLES

# this script was created by Tobias Meyer
# this script was created to take part in the Kaggle-competition as part of the EduX/MIT-MOOC "The Analytics Edge", offered in spring 2015
# try this for more detail: https://courses.edx.org/courses/MITx%2F15.071x_2%2F1T2015/info

# load data
NewsTrain = read.csv("NYTimesBlogTrain.csv", stringsAsFactors=FALSE)
NewsTest = read.csv("NYTimesBlogTest.csv", stringsAsFactors=FALSE)


###################### CENTRAL PREPREOCESSING IN A SINGLE DATAFRAME ######################

# put everything into a single dataframe - test data will have an outcome of NA
NewsTest$Popular = NA
News = rbind(NewsTrain, NewsTest)

# convert several variables to factors
News$NewsDesk = as.factor(News$NewsDesk)
News$SectionName = as.factor(News$SectionName)
News$SubsectionName = as.factor(News$SubsectionName)

# get information from the timestamp, get rid of timestamp itself afterwards
News$PubDate = strptime(News$PubDate, "%Y-%m-%d %H:%M:%S")
News$Hour = News$PubDate$hour
News$Weekday = News$PubDate$wday
News$Weekday = as.factor(News$Weekday)
# News$Morning = News$Hour > 6 & News$Hour <= 9
# News$Midday = News$Hour > 9 & News$Hour <= 17
# News$Evening = News$Hour >17 & News$Hour <= 22
# News$Night = ! News$Morning & ! News$Evening & ! News$Midday
# this is a bit ugly, but should do the bucketing hours alright. Will create 7 buckets, one from 0-6 and the others of 3 hours length
y = abs(News$Hour-1)
y = abs(y-1)
y = abs(y-1)
y = round(y/3)
News$Daytime = as.factor(y)
News$Hour = NULL
News$PubDate = NULL

News$LogWordCount = log(News$WordCount + 1)

################################ CREATE BAG OF WORDS ####################################
library(tm)

# for headlines
CorpusHeadline = Corpus(VectorSource(News$Headline))
CorpusHeadline = tm_map(CorpusHeadline, tolower)
CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)
CorpusHeadline = tm_map(CorpusHeadline, removePunctuation)
CorpusHeadline = tm_map(CorpusHeadline, removeWords, stopwords("english"))
CorpusHeadline = tm_map(CorpusHeadline, stemDocument)

headlineSparsity = 0.99 # words must be contained in at least 1% of documents
dtmHeadline = DocumentTermMatrix(CorpusHeadline)
sparseHeadline = removeSparseTerms(dtmHeadline, headlineSparsity)
dfHeadlineWords = as.data.frame(as.matrix(sparseHeadline))
colnames(dfHeadlineWords) = make.names(colnames(dfHeadlineWords))
colnames(dfHeadlineWords) = paste("head", colnames(dfHeadlineWords), sep=".")
# ncol(dfHeadlineWords)

# for abstract - we'll use only one of these (snippet/abstract), since they are very often identical. This is the case in 8273 of 8402 cases.
CorpusAbstract = Corpus(VectorSource(News$Abstract))
CorpusAbstract = tm_map(CorpusAbstract, tolower)
CorpusAbstract = tm_map(CorpusAbstract, PlainTextDocument)
CorpusAbstract = tm_map(CorpusAbstract, removePunctuation)
CorpusAbstract = tm_map(CorpusAbstract, removeWords, stopwords("english"))
CorpusAbstract = tm_map(CorpusAbstract, stemDocument)

abstractSparsity = 0.975
dtmAbstract = DocumentTermMatrix(CorpusAbstract)
sparseAbstract = removeSparseTerms(dtmAbstract, abstractSparsity)
dfAbstractWords = as.data.frame(as.matrix(sparseAbstract))
colnames(dfAbstractWords) = make.names(colnames(dfAbstractWords))
colnames(dfAbstractWords) = paste("abstr", colnames(dfAbstractWords), sep=".")

News = cbind(News, dfHeadlineWords, dfAbstractWords)
News$Headline = NULL
News$Snippet = NULL
News$Abstract = NULL

############################## SIMPLE LOGISTIC MODEL #####################################

Train = subset(News, ! is.na(News$Popular))
summary(Train)
Test = subset(News, is.na(News$Popular))
logModel = glm(Popular ~ . - UniqueID, data = Train, family=binomial)

library(ROCR)
pred_LM = predict(logModel, type="response")
predROCR_LM = prediction(pred_LM, Train$Popular)
table(Train$Popular, pred_LM > 0.5)
performance(predROCR_LM, "auc")@y.values
summary(logModel)

predTestLM = predict(logModel, newdata=Test, type="response")
SubmissionLogModel = data.frame(UniqueID = Test$UniqueID, Probability1 = predTestLM)
#write.csv(SubmissionLogModel, "SubmissionLogModelBagOfWordsDifferentFeautures.csv", row.names=FALSE)


########################### SIMPLE RANDOM FOREST MODEL ##################################

Train = subset(News, ! is.na(News$Popular))
Test = subset(News, is.na(News$Popular))
summary(Train)
library(randomForest)
rfModel = randomForest(Popular ~ ., data = Train, nodesize = 25, ntree = 1000)

library(ROCR)
pred_RF = predict(rfModel, type="response")
predROCR_RF = prediction(pred_RF, Train$Popular)
table(Train$Popular, pred_RF > 0.5)
performance(predROCR_RF, "auc")@y.values
summary(rfModel)

predTestRF = predict(rfModel, newdata=Test, type="response")
SubmissionRFModel = data.frame(UniqueID = Test$UniqueID, Probability1 = predTestRF)
#write.csv(SubmissionRFModel, "SubmissionRFBagOfWords.csv", row.names=FALSE)


##################################### HYBRID ############################################


predTrain = (pred_LM + pred_RF) / 2
predROCR = prediction(predTrain, Train$Popular)
table(Train$Popular, predTrain > 0.5)
performance(predROCR, "auc")@y.values

predTest = (predTestLM + predTestRF) / 2.0
SubmissionHybridModel = data.frame(UniqueID = Test$UniqueID, Probability1 = predTest)
write.csv(SubmissionHybridModel, "20150422_1_SubmissionHybridBagOfWordsDifferentFeatures.csv", row.names=FALSE)


############################# SOME TESTS ######################################

dfCheckResults = data.frame(UniqueID = Test$UniqueID, Probability1 = predTest, Recommended = (predTest > 0.5), NewsDesk = Test$NewsDesk)
summary(subset(dfCheckResults, dfCheckResults$NewsDesk == "Magazine"))
