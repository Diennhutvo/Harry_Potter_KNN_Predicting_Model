# Load data and package 

library(caret)
library(FNN)
library(class)

Lab1 <- read.csv("UniversalBank.csv",header = TRUE)

# Observing data 

head(Lab1,10)

str(Lab1)

Lab1$Personal.Loan <- as.factor(Lab1$Personal.Loan)
Lab1$Education <- as.factor(Lab1$Education)
Lab1$Securities.Account <- as.factor(Lab1$Securities.Account)
Lab1$CD.Account <- as.factor(Lab1$CD.Account)
Lab1$Online <- as.factor(Lab1$Online)
Lab1$CreditCard <- as.factor(Lab1$CreditCard)

# Removing unnecessary data

Lab1 <- Lab1[ , -c(1, 5)]
names(Lab1)

# Task 


## Training- Validation Split

set.seed(777)

train_index <- sample(1:nrow(Lab1), 0.6 * nrow(Lab1))
valid_index <- setdiff(1:nrow(Lab1), train_index)

train_df <- Lab1[train_index, ]
valid_df <- Lab1[valid_index, ]

## Observe new data 

nrow(train_df)

nrow(valid_df)

head(train_df)

t(t(names(Lab1)))

## Norm data 

train_norm <- train_df
valid_norm <- valid_df

norm_values <- preProcess(train_df[, -c(6, 8:12)],
                          method = c("center",
                                     "scale"))

# Train norm

train_norm[, -c(6, 8:12)] <- predict(norm_values,
                               train_df[, -c(6, 8:12)])

head(train_norm)


# Valid norm

valid_norm[, -c(6, 8:12)] <- predict(norm_values,
                               valid_df[, -c(6, 8:12)])

head(valid_norm)

# Add new customer 

person1 <- data.frame(Age = 40,
                        Experience = 10,
                        Income = 84,
                        Family = 2,
                        CCAvg = 2,
                        Education = 2,
                        Mortgage = 0,
                        Securities.Account = 0,
                        CD.Account = 0 , 
                        Online = 1,
                        CreditCard = 1)
person1


person1$Education <- as.factor(person1$Education)
person1$Securities.Account <- as.factor(person1$Securities.Account)
person1$CD.Account <- as.factor(person1$CD.Account)
person1$Online <- as.factor(person1$Online)
person1$CreditCard <- as.factor(person1$CreditCard)

# Predict Norm for person 1 

person1_norm <- predict(norm_values, person1)
person1_norm

# k = 3 

knn_pred_k3 <- class::knn(train = train_norm[,-c(8)], 
                          test = valid_norm [, -c(8)], 
                          cl = train_df$Personal.Loan, 
                          k = 3)

confusionMatrix(knn_pred_k3, as.factor(valid_df[, 8]))

# k = 5

knn_pred_k5 <- class::knn(train = train_norm[,-c(8)], 
                          test = valid_norm [, -c(8)], 
                          cl = train_df$Personal.Loan, 
                          k = 5)

confusionMatrix(knn_pred_k5, as.factor(valid_df[, 8]))

# k = 7


knn_pred_k7 <- class::knn(train = train_norm[,-c(8)], 
                          test = valid_norm [, -c(8)], 
                          cl = train_df$Personal.Loan, 
                          k = 7)

confusionMatrix(knn_pred_k7, as.factor(valid_df[, 8]))

# k = 9 


knn_pred_k9 <- class::knn(train = train_norm[,-c(8)], 
                          test = valid_norm [, -c(8)], 
                          cl = train_df$Personal.Loan, 
                          k = 9)

confusionMatrix(knn_pred_k9, as.factor(valid_df[, 8]))


# => K = 3 Has the best accuracy with 0.966

# Person 1 

person1_pred <- class::knn(train = train_norm[, -c(8)],
                             test = person1_norm,
                             cl = train_norm$Personal.Loan,
                             k = 3)

person1_pred

# Answer 

## They were all in numerous variable, thus, I converted them back as factor
## The best k from the given four is k= 3 since its accuracy is 0.966
## the person is not likely to accept the loan since prediction = 0


