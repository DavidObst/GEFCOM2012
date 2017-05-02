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
library(mgcv)
library(Probing)

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

dist <- function(x,y,type=2)
{
  R <- switch(type,
    "2"=sqrt(sum((x-y)^2)),
    "22"=sum((y-x)^2))
  
  return(R)
}

power.curve.est <- function(x,y,sep=10,n)
{
  intervalle_vec = seq(0,1,length.out = sep)
  power_points = rep(0,sep-1)
  
  for(k in 1:(sep-1))
  {
    indices = which((x>= intervalle_vec[k]) & (x < intervalle_vec[k+1]) )
    power_points[k] = median(y[indices])
    
  }
  
  power_points = c(0,power_points)
  
  power_curve = spline(intervalle_vec,y=power_points,xmin=0,xmax=1,n=n)
  
  return(power_curve)
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

##############################
wf1.new <- readRDS("data/wf1_new.RDS")

names <- paste("wf",1:7,sep="")

##date = moment when the FC are issued
##hors = hours of advance of the FC
##e.g.  if "date" = 2009070812 and "hors" = 1, the forecast is for the 8th of July 2009 at 13:00

################################
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
df.FC$month <- month(df.FC$date)

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

df.FC$ws.angle.B <- 0
df.FC$ws.angle.B.p1 <- 0
df.FC$ws.angle.B.p2 <- 0
df.FC$ws.angle.B.p3 <- 0
df.FC$ws.angle.B.n1 <- 0
df.FC$ws.angle.B.n2 <- 0
df.FC$ws.angle.B.n3 <- 0

df.FC$wp_hn01 <- 0 ## Previous know wp value for a given farm

df.FC$ws2 = (df.FC$ws)^2
df.FC$ws3 = (df.FC$ws)^3

## ws.angle construction by farm and by horizon ##
for(k in 1:nfarm)
{
  indices0 <- which((df.FC$farm==paste(k)) )
  
  for(l in 1:4)
  {
    indices <- which((df.FC$farm==paste(k)) & (df.FC$horizon.int == paste("(",(l-1)*12,";",l*12,"]",sep="")))
    
    indices.L <- sample(indices,size=floor(0.6*length(indices)) )
    
    gbm.ws.wd <- gbm(wp~wd_cut_2*(ws+ws2+ws3),data=df.FC[indices.L,],distribution="gaussian",n.trees=1000,
                     interaction.depth=3,n.minobsinnode=5,shrinkage =  0.01)
    
    ws.angle <- predict(gbm.ws.wd,n.trees=1000,newdata=df.FC[indices,])
    df.FC$ws.angle[indices] <- ws.angle
#     
#     df.FC$ws.angle.p1[indices] <- lag.generation(ws.angle,lag=4,other.values=sample(ws.angle,4))
#     df.FC$ws.angle.p2[indices] <- lag.generation(ws.angle,lag=8,other.values=sample(ws.angle,8))
#     df.FC$ws.angle.p3[indices] <- lag.generation(ws.angle,lag=12,other.values=sample(ws.angle,12))
#     
#     df.FC$ws.angle.n1[indices] <- lag.generation(ws.angle,lag=-4,other.values=sample(ws.angle,4))
#     df.FC$ws.angle.n2[indices] <- lag.generation(ws.angle,lag=-8,other.values=sample(ws.angle,8))
#     df.FC$ws.angle.n3[indices] <- lag.generation(ws.angle,lag=-12,other.values=sample(ws.angle,12))
  }
  
  df.FC$ws.angle.p1[indices0] <- lag.generation(df.FC$ws.angle[indices0],lag=4,other.values=sample(ws.angle,4))
  df.FC$ws.angle.p2[indices0] <- lag.generation(df.FC$ws.angle[indices0],lag=8,other.values=sample(ws.angle,8))
  df.FC$ws.angle.p3[indices0] <- lag.generation(df.FC$ws.angle[indices0],lag=12,other.values=sample(ws.angle,12))
  
  df.FC$ws.angle.n1[indices0] <- lag.generation(df.FC$ws.angle[indices0],lag=-4,other.values=sample(ws.angle,4))
  df.FC$ws.angle.n2[indices0] <- lag.generation(df.FC$ws.angle[indices0],lag=-8,other.values=sample(ws.angle,8))
  df.FC$ws.angle.n3[indices0] <- lag.generation(df.FC$ws.angle[indices0],lag=-12,other.values=sample(ws.angle,12))
  
  
}

## ws.angle construction 2 ##
for(k in 1:nfarm)
{
  indices <- which((df.FC$farm==paste(k)) )
  
  indices.L <- sample(indices,size=floor(0.6*length(indices)) )
  
  gbm.ws.wd <- gbm(wp~wd_cut_2*(ws+ws2+ws3),data=df.FC[indices.L,],distribution="gaussian",n.trees=1000,
                   interaction.depth=3,n.minobsinnode=5,shrinkage =  0.01)
  
  ws.angle.B <- predict(gbm.ws.wd,n.trees=1000,newdata=df.FC[indices,])
  df.FC$ws.angle.B[indices] <- ws.angle.B
  
  df.FC$ws.angle.B.p1[indices] <- lag.generation(ws.angle.B,lag=4,other.values=sample(ws.angle.B,4))
  df.FC$ws.angle.B.p2[indices] <- lag.generation(ws.angle.B,lag=8,other.values=sample(ws.angle.B,8))
  df.FC$ws.angle.B.p3[indices] <- lag.generation(ws.angle.B,lag=12,other.values=sample(ws.angle.B,12))
  
  df.FC$ws.angle.B.n1[indices] <- lag.generation(ws.angle.B,lag=-4,other.values=sample(ws.angle.B,4))
  df.FC$ws.angle.B.n2[indices] <- lag.generation(ws.angle.B,lag=-8,other.values=sample(ws.angle.B,8))
  df.FC$ws.angle.B.n3[indices] <- lag.generation(ws.angle.B,lag=-12,other.values=sample(ws.angle.B,12))
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

df.FC <- wf.list[[1]]

for(k in 2:7)
{
  df.FC <- rbind(df.FC,wf.list[[k]])
}


#############################################################

LL <- nrow(train)
dates.L <- ymd_h(train$date[1:floor(0.3*LL)])
dates.V <- ymd_h(train$date[floor(0.3*LL+1):floor(0.8*LL)])
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

ws.angle.splitted <- split(wf.list[[1]][,c(paste("ws.angle.p",1:3,sep=""),"ws.angle",paste("ws.angle.n",1:3,sep=""))],as.factor(wf.list[[1]]$horizon.int)) 

for(k in 1:4)
{
  wf1.new[,paste(colnames(ws.angle.splitted[[k]]),12*k,sep=".")] <- ws.angle.splitted[[k]]
}


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

### Version B avec ws.angle construit juste pour chaque ferme ###

## Model 1B ##
model1B <-  gbm(
  formula =  wp ~ 
    ws + wd_cut +
    ws.angle.B + 
    ws.angle.B.p1 + ws.angle.B.p2 + ws.angle.B.p3 + 
    ws.angle.B.n1 + ws.angle.B.n2 + ws.angle.B.n3 +
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

## Model 2B ##
model2B <-  gbm(
  formula =  wp ~ ws + farm +
    ws.angle.B + ws.angle.B.p1 + ws.angle.B.p2 +
    ws.angle.B.p3 + ws.angle.B.n1 + ws.angle.B.n2 +
    ws.angle.B.n3 + hour + month  + dist +wp_hn01,
  data = learning.1,
  n.trees = 3500,
  interaction.depth = 8,
  n.minobsinnode = 30,
  shrinkage =  0.05,
  distribution = "gaussian",
  train.fraction = 0.8,
  keep.data = F
)

## Model 3B ##
model3B <-  gbm(
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


####################################################################
####################################################################

lm.data.learn = data.frame("wp"=learning.1$wp,"y.1"=predict(model1,newdata=learning.1,n.trees=3500),
                           "y.2"=predict(model2,newdata=learning.1,n.trees=3500),
                           "y.3"=predict(model3,newdata=learning.1,n.trees=3500))

lm.data.valid = data.frame("wp"=valid.1$wp,"y.1"=predict(model1,newdata=valid.1,n.trees=3500),
                           "y.2"=predict(model2,newdata=valid.1,n.trees=3500),
                           "y.3"=predict(model3,newdata=valid.1,n.trees=3500))

lm.ensemble = lm(wp ~ ., data=lm.data.learn)


##############################
### Type B  ##
##############

lm.data.learn.B = data.frame("wp"=learning.1$wp,"y.1"=predict(model1B,newdata=learning.1,n.trees=3500),
                           "y.2"=predict(model2B,newdata=learning.1,n.trees=3500),
                           "y.3"=predict(model3B,newdata=learning.1,n.trees=3500))

lm.data.valid.B = data.frame("wp"=valid.1$wp,"y.1"=predict(model1B,newdata=valid.1,n.trees=3500),
                           "y.2"=predict(model2B,newdata=valid.1,n.trees=3500),
                           "y.3"=predict(model3B,newdata=valid.1,n.trees=3500))

lm.ensemble.B = lm(wp ~ ., data=lm.data.learn.B)

##############################

online.estimation <- function(n_ahead,data,list.models,lm.model)
{
  y.hat <- rep(0,n_ahead)

  X.online <- data[1:4,]
  
  ########################################
  
  for(t in 1:n_ahead)
  {
    tt <-  (4*(t-1)+1):(4*t)
    tt1 <- (4*(t)+1):(4*(t+1))
    y.t1.1 <- predict(list.models[[1]],newdata=X.online[tt,],n.trees=3500)
    y.t1.2 <- predict(list.models[[2]],newdata=X.online[tt,],n.trees=3500)
    y.t1.3 <- predict(list.models[[3]],newdata=X.online[tt,],n.trees=3500)
   
    
    y.t1 <- predict(lm.model,newdata=data.frame("y.1"=y.t1.1,"y.2"=y.t1.2,"y.3"=y.t1.3))
    
    y.hat[t] <- mean(as.numeric(y.t1))
  
    X.online <- rbind(X.online,data[tt1,])
    
    for(j in 2:4)
    {
      X.online[tt1,paste("wp_hn0",j,sep="")] <- X.online[tt,paste("wp_hn0",j-1,sep="")]
    }
    
    X.online$wp_hn01[tt1] <- y.hat[t]
    
    
  }
  
  to.return <- list("y.hat"=y.hat,"X.online"=X.online)
  
}
################################################
#################################################

n_ahead = 48

init.date <- ymd_h(2010112912)
init.index <- which(eval.1$date == init.date)[1]

data = eval.1[init.index+(1:(4*n_ahead))-1,]

y = by(data$wp,as.factor(data$date),mean)

test1 = online.estimation(n_ahead,data,list(model1,model2,model3),lm.ensemble)
test1B = online.estimation(n_ahead,data,list(model1B,model2B,model3B),lm.ensemble)

rmse(y,test1$y.hat)
mae(y,test1$y.hat)
amape(y,test1$y.hat)


rmse(y,test1B$y.hat)
mae(y,test1B$y.hat)
amape(y,test1B$y.hat)

################################################
### OADMM ####
##############

learning.OADMM <- train[which(ymd_h(train$date) %in% dates.L),]
valid.OADMM <- train[which(ymd_h(train$date) %in% dates.V),]
eval.OADMM <- train[which(ymd_h(train$date) %in% dates.E),]

OADMM.data <- OADMM.prep(eval.OADMM[,-1],l=2,1,1:7)
OADMM.est <- OADMM(OADMM.data$X,OADMM.data$y,0.2,1,2,sigmoid.coeffs=c(0,1,1),maxit=nrow(OADMM.data$X))

plot(y,type="l")
lines(test1$y.hat,col="blue")
lines(test1B$y.hat,col="red")
lines(OADMM.est$y.chap[(init.index.2:(init.index.2+47))-2],col="red")
