#======================================================================================================
#======================================================================================================
library(leaps)
library(caret)
library(elasticnet)
library(car)
new_songs <- read.csv('isye 6414/project/train_df_v2.csv')
new_songs <- new_songs[,2:16] #dropping song id/name



#Baseline multivariate linear regression - without scaling
baseline_model <- lm(Popularity ~., data=new_songs)
summary(baseline_model)


train_control = trainControl(method='cv',number=10)
new_baseline_model_with_cv <- train(Popularity ~ ., data=new_songs, method='lm',trControl=train_control)
print(new_baseline_model_with_cv)
summary(new_baseline_model_with_cv) #much better numbers than before

#Baseline model with columns scaled from 0 - 1
scale <- function(x){(x-min(x))/(max(x)-min(x))}
new_scaled_songs <- new_songs
new_scaled_songs$loudness = scale(new_songs$loudness)
new_scaled_songs$tempo = scale(new_songs$tempo)
new_scaled_songs$duration_ms = scale(new_songs$duration_ms)
new_scaled_songs$key = scale(new_songs$key)
new_scaled_songs$time_signature = scale(new_songs$time_signature)
new_scaled_songs$Year = scale(new_songs$Year)
new_scaled_songs$Popularity = scale(new_songs$Popularity)

#new baseline scaled model
new_scaled_baseline_model <- lm(Popularity ~., data=new_scaled_songs)
summary(new_scaled_baseline_model)

new_scaled_model <- train(Popularity ~ ., data=new_scaled_songs, method='lm',trControl=train_control)
print(new_scaled_model)
summary(new_scaled_model)

#Stepwise model
new_stepwise_model <- train(Popularity ~ ., data=new_songs, method='leapSeq',trControl=train_control)
print(new_stepwise_model)
summary(new_stepwise_model)

#Test stepwise regression with more predictors - see which ones are consistently most important
new_stepwise_model_more_predictors <- regsubsets(Popularity ~ ., data = new_scaled_songs, nvmax = 6)
print(new_stepwise_model_more_predictors)
summary(new_stepwise_model_more_predictors)


#Lasso
new_lasso_model <- train(Popularity ~ ., data=new_scaled_songs, method='lasso',trControl=train_control)
print(new_lasso_model)
summary(new_lasso_model)

#Conduct cross validation on the model using the 6 important features and the 3 interaction terms with OLS
new_model <- train(Popularity ~ energy+loudness+acousticness+instrumentalness+valence+Year+energy*loudness+instrumentalness*valence+Year*instrumentalness, data=new_songs, method='lm',trControl=train_control)
print(new_model)
summary(new_model)

#Conduct cross validation on the model using only the 6 important features with OLS
new_model <- train(Popularity ~ energy+acousticness+instrumentalness+valence+Year, data=new_songs, method='lm',trControl=train_control)
print(new_model)
summary(new_model)




#Read in the test dataset to test prediction accuracy
test_df <- read.csv('isye 6414/project/test_df_v2.csv')

#Test accuracy of predictions of models using the test data
pred <- predict(new_model, test_df)
baseline_pred <- predict(baseline_model,test_df)

sqrt(mean((test_df$Popularity - pred)^2))
mean(abs(test_df$Popularity - pred))

sqrt(mean((test_df$Popularity - baseline_pred)^2))
mean(abs(test_df$Popularity - baseline_pred))




#Create all the boxcox plots - did each predictor variable individually as well as all of them together
#Each plot showed that no transformation on the response variable is necessary though
library(MASS)
boxcox(new_songs$Popularity ~ new_songs$Year)
boxcox(new_songs$Popularity ~ new_songs$energy)
boxcox(new_songs$Popularity ~ new_songs$acousticness)
boxcox(new_songs$Popularity ~ new_songs$instrumentalness)
boxcox(new_songs$Popularity ~ new_songs$valence)
boxcox(new_songs$Popularity ~ new_songs$Year + new_songs$energy + new_songs$acousticness + new_songs$instrumentalness + new_songs$valence + new_songs$loudness)




#Weighted Least Squares Implementation
#Using iteratively reweighted least squares fitting algorithm
modelsd <- lm(abs(resid(new_model)) ~ new_songs$energy + new_songs$loudness + new_songs$acousticness + new_songs$instrumentalness + new_songs$valence + new_songs$Year)
summary(modelsd)

fitted(modelsd)

#Part 2
w <- 1 / (fitted(modelsd)^2)
modelwls <- lm(Popularity ~ energy+acousticness+instrumentalness+valence+Year, data=new_songs, weight = w)
summary(modelwls)


#Part 3
modelsd3 <- lm(abs(resid(modelwls)) ~ new_songs$energy + new_songs$loudness + new_songs$acousticness + new_songs$instrumentalness + new_songs$valence + new_songs$Year)

w2 <- 1 / (fitted(modelsd3)^2)
modelwls2 <- lm(Popularity ~ energy+acousticness+instrumentalness+valence+Year, data=new_songs, weight = w2)
summary(modelwls2)

#Repeat
modelsd4 <- lm(abs(resid(modelwls2)) ~ new_songs$energy + new_songs$loudness + new_songs$acousticness + new_songs$instrumentalness + new_songs$valence + new_songs$Year)
w3 <- 1 / (fitted(modelsd4)^2)
modelwls3 <- lm(Popularity ~ energy+acousticness+instrumentalness+valence+Year, data=new_songs, weight = w2)
summary(modelwls3)

wls_pred <- predict(modelwls3, test_df)

sqrt(mean((test_df$Popularity - wls_pred)^2))
mean(abs(test_df$Popularity - wls_pred))

