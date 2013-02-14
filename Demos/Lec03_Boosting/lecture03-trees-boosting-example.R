library('gbm')
library(randomForest)

# create data
N <- 1000
X1 <- 4*runif(N)
X2 <- 2*runif(N)
X3 <- ordered(sample(letters[1:4],N,replace=TRUE),levels=letters[4:1])
X4 <- factor(sample(letters[1:6],N,replace=TRUE))
X5 <- factor(sample(letters[1:3],N,replace=TRUE))
X6 <- 3*runif(N)
X=cbind(X1,X2,X3,X4,X5,X6)

mu <- c(-1,0,1,2)[as.numeric(X3)]
SNR <- 10 # signal-to-noise ratio
Y <- X1**1.5 + 2 * (X2**.5) + X1*X2*X6 + mu
sigma <- sqrt(var(Y)/SNR)
Y <- Y + rnorm(N,0,sigma)

ind=sample(N,floor(N*2/3))

trn.x=X[ind,]
trn.y=Y[ind]

tst.x=X[-ind,]
tst.y=Y[-ind]
  
# linear model
model.lm = lm(trn.y~ ., data=data.frame(trn.x))
y.lm = predict(model.lm,data.frame(tst.x))

# decision tree
model.dt = gbm.fit(trn.x,trn.y,distribution="gaussian",shrinkage=1,bag.fraction=1,n.trees=1,interaction.depth=10)
y.dt = predict.gbm(model.dt,tst.x,1)

# gradient boosting
model.gbm = gbm.fit(trn.x,trn.y,distribution="gaussian",shrinkage=0.1,bag.fraction=1,n.trees=100,interaction.depth=10)
y.gbm = predict.gbm(model.gbm,tst.x,100)

# random forest
model.rf = randomForest(trn.x,trn.y, do.trace=TRUE,importance=TRUE,samplesize=N, ntree = 100)
y.rf = predict(model.rf,tst.x)


cat(paste("linear model prediction error=",mean((tst.y-y.lm)^2),'\n')) 
cat(paste("decision tree prediction error=",mean((tst.y-y.dt)^2),'\n')) 
cat(paste("gradient boosting prediction error=",mean((tst.y-y.gbm)^2),'\n')) 
cat(paste("random forest prediction error=",mean((tst.y-y.rf)^2),'\n'))

