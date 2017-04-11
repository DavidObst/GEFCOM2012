rm(list=objects())
graphics.off()

###############################

library(lubridate)
library(gbm)
library(kknn)
library(neuralnet)
library(dplyr)
library(DAAG)
library(caret)
library(Utility)
library(flexclust)

###############################
## useful functions ##
######################


relu <- function(x)
{
  if(is.na(x)) {R <- NA}
  else if(x<0) {R <- 0}
  else{R <- x}
  return(R)
}

wd.cut.func <- function(wd,splitting)
{
  if(is.na(wd)) {R <- NA}
  k <- relu( floor((wd-1e-5)/splitting) )
  R <- paste("(",k*splitting,";",(k+1)*splitting,"]",sep="")
  return(R)
}

pred.cut.func <- function(hors,splitting)
{
  if(is.na(hors)) {R <- NA}
  k <- relu( floor((hors-1e-5)/splitting) )
  R <- paste("(",k*splitting,";",(k+1)*splitting,"]",sep="")
  return(R)
}

###############################

train<- read.csv("data/train.csv",sep=",",dec=".",header=T)
benchmark <- read.csv("data/benchmark.csv",sep=",",dec=".",header=T)
wf1 <- read.csv("data/windforecasts_wf1.csv",sep=",",dec=".",header=T)
wf2 <- read.csv("data/windforecasts_wf2.csv",sep=",",dec=".",header=T)
wf3 <- read.csv("data/windforecasts_wf3.csv",sep=",",dec=".",header=T)
wf4 <- read.csv("data/windforecasts_wf4.csv",sep=",",dec=".",header=T)
wf5 <- read.csv("data/windforecasts_wf5.csv",sep=",",dec=".",header=T)
wf6 <- read.csv("data/windforecasts_wf6.csv",sep=",",dec=".",header=T)
wf7 <- read.csv("data/windforecasts_wf7.csv",sep=",",dec=".",header=T)

names <- paste("wf",1:7,sep="")

##date = moment when the FC are issued
##hors = hours of advance of the FC
##e.g.  if "date" = 2009070812 and "hors" = 1, the forecast is for the 8th of July 2009 at 13:00

################################
## A part of the set corresponding the data on which evaluation would be made for GEFCOM will be used
## to train the gbm model to obtain ws.angle 

# indices.ws.angle <- which( (ymd_h(wf1$date) %in% ymd_h(benchmark$date)) & !is.na(wf1$ws)  )
# indices.ws.angle.2 <- which( ymd_h(benchmark$date) %in% ymd_h(wf1$date) )
# 
# list.ws.angle <- list()
# 
# for(k in 1:7)
# {
#   list.ws.angle[[k]] <- get(names[k])[indices.ws.angle,]
#   list.ws.angle[[k]]$horizon.int <-  sapply(list.ws.angle[[k]]$hors,pred.cut.func,splitting=12)
#   list.ws.angle[[k]]$start <- ymd_h(list.ws.angle[[k]]$date)
#   list.ws.angle[[k]]$date <- ymd_h(list.ws.angle[[k]]$date)+hours(list.ws.angle[[k]]$hors)
#   list.ws.angle[[k]] <- arrange(list.ws.angle[[k]],date)
#   list.ws.angle[[k]]$wp <- rep(benchmark[,paste("wp",k,sep="")],each=4)
# }


################################

end.training <- ymd_h(2010123123)

to.remove <- which((ymd_h(wf1$date)+hours(wf1$hors) < ymd_h(2009070213)) | ((ymd_h(wf1$date)+hours(wf1$hors) > end.training)))
to.remove.2 <- which((ymd_h(train$date) < ymd_h(2009070213)) | (ymd_h(train$date) > end.training) )

train <- train[-to.remove.2,]
wf1 <- wf1[-to.remove,]
wf2 <- wf2[-to.remove,]
wf3 <- wf3[-to.remove,]
wf4 <- wf4[-to.remove,]
wf5 <- wf5[-to.remove,]
wf6 <- wf6[-to.remove,]
wf7 <- wf7[-to.remove,]

##############################################
####" Feature creation ######
#############################

nfarm <- 7

df <- data.frame("wp"=rep(0,nfarm*nrow(train)),"date"=rep(0,nfarm*nrow(train)),"farm"=rep(0,nfarm*nrow(train)))

for(k in 1:nfarm)
{
  kk <- (k-1)*nrow(train) + 1
  kk.1 <- k*nrow(train)
  
  df$wp[kk:kk.1] <- train[,paste("wp",k,sep="")]
  df$farm[kk:kk.1] <- k
  df$date[kk:kk.1] <- train$date
}

## Calendar variables
df$date <- ymd_h(df$date)
df$farm <- as.factor(df$farm)
df$hour <- as.factor(hour(df$date))
df$month <- as.factor(month(df$date))
df$year <- as.factor(year(df$date))

## Rearranging the data frame by instant of the wp measurement
df <- arrange(df,date)

##Forecast features

df.FC <- data.frame("date"=rep(0,nfarm*nrow(wf1)),"farm"=rep(0,nfarm*nrow(wf1)),"start"=rep(0,nfarm*nrow(wf1)),
                    "dist"=rep(0,nfarm*nrow(wf1)),"turn"=rep(0,nfarm*nrow(wf1)),"ws"=rep(0,nfarm*nrow(wf1)),
                    "wd"=rep(0,nfarm*nrow(wf1)),"wd_cut"=rep(0,nfarm*nrow(wf1)) )


for(k in 1:nfarm)
{
  kk <- (k-1)*nrow(wf1) +1
  kk.1 <- k*nrow(wf1)
  
  df.FC$date <- get(names[k])$date ## Not finished - must convert to posixct
  df.FC$start[kk:kk.1] <- get(names[k])$date ## the same
  df.FC$farm[kk:kk.1] <- k
  df.FC$dist[kk:kk.1] <-  get(names[k])$hors
  df.FC$turn[kk:kk.1] <- get(names[k])$date ##same
  ## set useless in our case
  df.FC$ws[kk:kk.1] <- get(names[k])$ws
  df.FC$wd[kk:kk.1] <- get(names[k])$wd
  
}

df.FC$date <- ymd_h(df.FC$date) + hours(df.FC$dist)
df.FC$start <- ymd_h(df.FC$start)
df.FC$turn <- as.factor(hour(df.FC$start))
df.FC$wd_cut <- as.factor(sapply(df.FC$wd,wd.cut.func,splitting=30))
df.FC$wd_cut_2 <- as.factor(sapply(df.FC$wd,wd.cut.func,splitting=8))

df.FC$hour <- hour(df.FC$date)
df.FC$month <- hour(df.FC$date)

df.FC <- arrange(df.FC,date)

df.FC$wp <- 0

for(k in 1:nfarm)
{
  indices <- which(df.FC$farm == paste(k))
  
  df.FC$wp[indices] <- rep(train[,k+1],each=4)
}

########################

df.FC[,"horizon.int"] <- sapply(df.FC$dist,pred.cut.func,splitting=12)

################################
### ws.angle feature creation ##
################################

df.FC$ws.angle <- 0
df.FC$ws.angle.p1 <- 0
df.FC$ws.angle.p2 <- 0
df.FC$ws.angle.p3 <- 0
df.FC$ws.angle.n1 <- 0
df.FC$ws.angle.n2 <- 0
df.FC$ws.angle.n3 <- 0

df.FC$wp_hn01 <- 0 ## Previous know wp value for a given farm

df.FC$ws2 = (df.FC$ws)^2
df.FC$ws3 = (df.FC$ws)^3


for(k in 1:nfarm)
{
  indices <- which(df.FC$farm==paste(k))
  
  indices.L <- 1:floor(0.6*length(indices))
  
  gbm.ws.wd <- gbm(wp~wd_cut_2*(ws+ws2+ws3),data=df.FC[indices.L,],distribution="gaussian",n.trees=1000,
                   interaction.depth=3,n.minobsinnode=5,shrinkage =  0.01)
  
  ws.angle <- predict(gbm.ws.wd,n.trees=1000,newdata=df.FC[indices,])
  df.FC$ws.angle[indices] <- ws.angle
  
  df.FC$ws.angle.p1[indices] <- lag.generation(ws.angle,lag=4,other.values=sample(ws.angle,4))
  df.FC$ws.angle.p2[indices] <- lag.generation(ws.angle,lag=8,other.values=sample(ws.angle,8))
  df.FC$ws.angle.p3[indices] <- lag.generation(ws.angle,lag=12,other.values=sample(ws.angle,12))
  
  df.FC$ws.angle.n1[indices] <- lag.generation(ws.angle,lag=4,other.values=sample(ws.angle,4))
  df.FC$ws.angle.n2[indices] <- lag.generation(ws.angle,lag=8,other.values=sample(ws.angle,8))
  df.FC$ws.angle.n3[indices] <- lag.generation(ws.angle,lag=12,other.values=sample(ws.angle,12))
  
  
}

df.FC$dist.quarter <- floor((df.FC$dist-1)/12)+1 ##same as 'horizon.int' 

###########################################################
wf.list <- split(df.FC,df.FC$farm)

for(k in 1:7)
{
  
  for(p in 1:4)
  {
    wf.list[[k]][,paste("wp_hn0",p,sep="")] <- lag.generation(wf.list[[k]]$wp,lag=4*p,rep(sample(wf.list[[k]]$wp,p),each=4))
    
  }
  
}



#############################################################

LL <- nrow(train)
dates.L <- ymd_h(train$date[1:floor(0.6*LL)])
dates.V <- ymd_h(train$date[floor(0.6*LL+1):floor(0.8*LL)])
dates.E <- ymd_h(train$date[floor(0.8*LL+1):LL])

## Learning indices ##

dates.L.2 <- wf.list[[1]]$date 
dates.V.2 <- wf.list[[1]]$date 
dates.E.2 <-wf.list[[1]]$date

### Clustering by farms ###

for(k in 1:7)
{
  clustering.by.farm <- kcca(wf.list[[k]]$wp[which(dates.L.2 %in% dates.L)],k=6)
  wf.list[[k]]$cluster.farm <- predict(clustering.by.farm,newdata=wf.list[[k]]$wp) + (k-1)*6
}

################################################################


df.FC <- wf.list[[1]]

for(k in 2:7)
{
  df.FC <- rbind(df.FC,wf.list[[k]])
}

################################################################


## Training for a single farm ##

learning.1 <- wf.list[[1]][which(dates.L.2 %in% dates.L),]
valid.1 <- wf.list[[1]][which(dates.V.2 %in% dates.V),]
eval.1 <- wf.list[[1]][which(dates.E.2 %in% dates.E),]


## Model 1 ##
model1 <-  gbm(
  formula =  wp ~ 
    ws + wd_cut +
    ws.angle + 
    ws.angle.p1 + ws.angle.p2 + ws.angle.p3 + 
    ws.angle.n1 + ws.angle.n2 + ws.angle.n3 +
    hour + 
    farm + dist,
  data = learning.1,
  n.trees = 3500,
  interaction.depth = 8,
  n.minobsinnode = 30,
  shrinkage =  0.05,
  distribution = "gaussian",
  train.fraction = 0.8,
  keep.data = F
)

## Model 2 ##
model2 <-  gbm(
  formula =  wp ~ ws + farm +
    ws.angle + ws.angle.p1 + ws.angle.p2 +
    ws.angle.p3 + ws.angle.n1 + ws.angle.n2 +
    ws.angle.n3 + hour + month  + dist +wp_hn01,
  data = learning.1,
  n.trees = 3500,
  interaction.depth = 8,
  n.minobsinnode = 30,
  shrinkage =  0.05,
  distribution = "gaussian",
  train.fraction = 0.8,
  keep.data = F
)

## Model 3 ##
model3 <-  gbm(
  formula =  wp ~ 
    farm + dist + wp_hn02 +
    wp_hn03 + wp_hn04 + hour + month +
    cluster.farm,
  data = learning.1,
  n.trees = 3500,
  interaction.depth = 8,
  n.minobsinnode = 30,
  shrinkage =  0.05,
  distribution = "gaussian",
  train.fraction = 0.8,
  keep.data = F
)

#############################################
########### Model for the whole set ####################

learning.c <- df.FC[which(df.FC$date %in% dates.L),]
valid.c <-df.FC[which(df.FC$date %in% dates.V),]
eval.c <- df.FC[which(df.FC$date %in% dates.E),]

## Model 1 ##
model1c <-  gbm(
  formula =  wp ~ 
    ws + wd_cut +
    ws.angle + 
    ws.angle.p1 + ws.angle.p2 + ws.angle.p3 + 
    ws.angle.n1 + ws.angle.n2 + ws.angle.n3 +
    hour + 
    farm + dist,
  data = learning.c,
  n.trees = 3500,
  interaction.depth = 8,
  n.minobsinnode = 30,
  shrinkage =  0.05,
  distribution = "gaussian",
  train.fraction = 0.8,
  keep.data = F
)

## Model 2 ##
model2c <-  gbm(
  formula =  wp ~ ws + farm +
    ws.angle + ws.angle.p1 + ws.angle.p2 +
    ws.angle.p3 + ws.angle.n1 + ws.angle.n2 +
    ws.angle.n3 + hour + month  + dist +wp_hn01,
  data = learning.c,
  n.trees = 3500,
  interaction.depth = 8,
  n.minobsinnode = 30,
  shrinkage =  0.05,
  distribution = "gaussian",
  train.fraction = 0.8,
  keep.data = F
)

## Model 3 ##
model3c <-  gbm(
  formula =  wp ~ 
    farm + dist + wp_hn02 +
    wp_hn03 + wp_hn04 + hour + month +
    cluster.farm,
  data = learning.c,
  n.trees = 3500,
  interaction.depth = 8,
  n.minobsinnode = 30,
  shrinkage =  0.05,
  distribution = "gaussian",
  train.fraction = 0.8,
  keep.data = F
)

####################################################################

lm.data.learn = data.frame("wp"=learning.1$wp,"y.1"=predict(model1,newdata=learning.1,n.trees=3500),
                           "y.2"=predict(model2,newdata=learning.1,n.trees=3500),
                           "y.3"=predict(model3,newdata=learning.1,n.trees=3500))
                           # "y.1c"=predict(model1c,newdata=learning.c[which(learning.c$farm==1),],n.trees=3500),
                          #  "y.2c"=predict(model2c,newdata=learning.c[which(learning.c$farm==1),],n.trees=3500),
                           # "y.3c"=predict(model3c,newdata=learning.c[which(learning.c$farm==1),],n.trees=3500)) 

lm.ensemble = lm(wp ~ ., data=lm.data.learn)

##################################################################
## ensemble for each horizon by lm ##

learning.weights <- split(predict(model1,n.trees=3500,newdata=learning.1),learning.1$horizon.int)
learning.weights  <- cbind(train$wp1[which(ymd_h(train$date) %in% dates.L)],as.data.frame(learning.weights))
colnames(learning.weights ) <- c("wp","h12","h24","h36","h48")
lm.weights  <- lm(wp~.,data=learning.weights )

weights.FC <- as.numeric(lm.weights $coefficients)

##################################################################

n_ahead <- 48
y.hat <- rep(0,n_ahead)

#init.date <- ymd_h(2010091312)
init.date <- ymd_h(2010091812)
init.index <- which(eval.1$date == init.date)[1]

X.online <- eval.1[init.index:(init.index+3),]

for(t in 1:n_ahead)
{
  tt <-  (4*(t-1)+1):(4*t)
  tt1 <- (4*(t)+1):(4*(t+1))
  y.t1.1 <- predict(model1,newdata=X.online[tt,],n.trees=3500)
  y.t1.2 <- predict(model2,newdata=X.online[tt,],n.trees=3500)
  y.t1.3 <- predict(model3,newdata=X.online[tt,],n.trees=3500)
  
#     y.t1.c1 <- predict(model1c,newdata=X.online[tt,],n.trees=3500)
#     y.t1.c2 <- predict(model2c,newdata=X.online[tt,],n.trees=3500)
#    y.t1.c3 <- predict(model3c,newdata=X.online[tt,],n.trees=3500)
  
  y.t1 <- predict(lm.ensemble,newdata=data.frame("y.1"=y.t1.1,"y.2"=y.t1.2,"y.3"=y.t1.3))
                                               # "y.1c"=y.t1.c1,"y.2c"=y.t1.c2,"y.3c"=y.t1.c3))
  

  y.hat[t] <- mean(as.numeric(y.t1))
  #y.hat[t] <- weights.FC[1] + rev(weights.FC[-1])%.%y.t1
  
  X.online <- rbind(X.online,eval.1[init.index + tt1,])

  for(j in 2:4)
  {
    X.online[tt1,paste("wp_hn0",j,sep="")] <- X.online[tt,paste("wp_hn0",j-1,sep="")]
  }
  
  X.online$wp_hn01[tt1] <- y.hat[t]
  
  
}

y = by(eval.1$wp[which(eval.1$date == init.date)[1]:(which(eval.1$date == init.date)[1]+4*n_ahead-1)],
       as.factor(eval.1$date[1:(4*n_ahead)]),mean)
y.persist = persistence.est(y,1,initial.values = learning.1$wp[nrow(learning.1)])

plot(y,type="l")
lines(y.hat,col="red")
lines(y.persist,col="blue")

amape(y,y.hat)
amape(y,y.persist)
mae(y,y.hat)


