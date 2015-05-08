# Parameters:
# data - data frame containing your training data
# model - the function to call to make the model (e.g. rpart, gbm, knn3 etc.)
# pargs - a list of arguments for the predict function
# ... - parameters for the model
# Examples:
# kfold(TrainData,randomForest,pargs=list(type="prob"),nodesize=1,ntree=500)
# kfold(TrainData,rpart,method="class",cp=0.01)
kfold = function(data,model,pargs=NULL,...) {
require(caret)
# k is the number of folds, 5 or 10 are common choices
k = 10
metrics = list(k)
means = numeric(10)
folds = createFolds(data$Popular,k=k,list=TRUE,returnTrain=TRUE)
for (i in 1:length(folds)) {
dfT = data[folds[[i]],]
dfP = data[folds[[i]]*-1,]
# change the formula (Popular~.) in the line below if you need to
m = model(Popular~.,data=dfT,...)
p = do.call(predict,c(list(object=m,newdata=dfP),pargs))
if (!is.vector(p))
if (ncol(p)>1)
p = p[,2]
mk = conf.matrix(dfP$Popular,p,0.5)$metrics
# remove the # in front of the print statement if you want to see 
# data for each iteration
print(mk)
metrics[[i]] = mk
}
for (i in 1:length(metrics)) {
means = means + metrics[[i]]
}
means = means / k
names(means) = names(metrics[[1]])
list(means=means)
}

conf.matrix = function (outcomes,predictions,cutoff) {
require(ROCR)
if (class(predictions) == "factor") {
auc=NA
t = table(outcomes,predictions)
}
else {
auc = performance(prediction(predictions,outcomes),"auc")@y.values[[1]]
t = table(outcomes,predictions=predictions >= cutoff)
if (cutoff > max(predictions)) {
t = cbind(t,"TRUE"=c(0,0))
}
else if (min(predictions) > cutoff)  {
t = cbind("FALSE"=c(0,0),t)
}
}
acc = (t[1,1] + t[2,2]) / length(predictions)
sen = t[2,2] / (t[2,1] + t[2,2])
spec = t[1,1] / (t[1,1] + t[1,2])
baseline.accuracy = max(t[1,1]+t[1,2],t[2,1]+t[2,2])/length(predictions)
list(confusion.matrix=t,metrics=c(sensitivity=sen,specificity=spec,accuracy=acc,baseline.acc=baseline.accuracy,auc=auc))
}

kfold(nytTrain3,randomForest,pargs=list(type="prob"),nodesize=25,ntree=5000)
