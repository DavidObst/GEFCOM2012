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
  
  indices.L <- sample(indices,size=floor(0.6*length(indices)) )
  
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

learning.new <- wf1.new[which(wf1.new$date %in% dates.L),]
valid.new <- wf1.new[which(wf1.new$date %in% dates.V),]
eval.new <- wf1.new[which(wf1.new$date %in% dates.E),]


####################################################################

lm.data.learn = data.frame("wp"=learning.1$wp,"y.1"=predict(model1,newdata=learning.1,n.trees=3500),
                           "y.2"=predict(model2,newdata=learning.1,n.trees=3500),
                           "y.3"=predict(model3,newdata=learning.1,n.trees=3500))
#"y.1c"=predict(model1c,newdata=learning.c[which(learning.c$farm==1),],n.trees=3500),
#"y.2c"=predict(model2c,newdata=learning.c[which(learning.c$farm==1),],n.trees=3500),
#"y.3c"=predict(model3c,newdata=learning.c[which(learning.c$farm==1),],n.trees=3500)) 

lm.data.valid = data.frame("wp"=valid.1$wp,"y.1"=predict(model1,newdata=valid.1,n.trees=3500),
                           "y.2"=predict(model2,newdata=valid.1,n.trees=3500),
                           "y.3"=predict(model3,newdata=valid.1,n.trees=3500))
#"y.1c"=predict(model1c,newdata=learning.c[which(learning.c$farm==1),],n.trees=3500),
#"y.2c"=predict(model2c,newdata=learning.c[which(learning.c$farm==1),],n.trees=3500),
#"y.3c"=predict(model3c,newdata=learning.c[which(learning.c$farm==1),],n.trees=3500))

lm.ensemble = lm(wp ~ ., data=lm.data.learn)

##################################################################
# 
# cluster.generation <- function(y.hat,y,data,n_clusters,threshold,m.gam,alpha=1,risk=0.1)
# {
#   sel.features <- list()
#   list.AWPF <- list()
#   list.delta <- list()
#   
#   indices.abnormal <- which(abs(y.hat - y) > threshold)
#   abnormal.data <- data[indices.abnormal,]
#   abnormal.tot <- abnormal.data
#   
#   m.cluster <- kcca(abnormal.data,k=n_clusters)
#   abnormal.tot$cluster <- m.cluster@second
#     
#   
#   radii <- rep(0,n_clusters)
#   radii.mean <- rep(0,n_clusters)
#   
#   #abnormal.data$date <- rep(0,nrow(abnormal.data))
#   
#   for(l in 1:n_clusters)
#   {
#     indices <- which(m.cluster@second==l)
#     distances <- apply(abnormal.data[indices,],1,dist,y=m.cluster@centers[l,],type=2)
#     radii[l] <- alpha*max(distances)
#     radii.mean[l] <- mean(distances)
#     
#     #########################
#     ### Correction engine ###
#     #########################
#     
#     delta <-  y[indices.abnormal[indices]]-y.hat[indices.abnormal[indices]]
#     list.delta[[l]] <- delta
#     
#     abnormal.tot[indices,"delta"] <- delta
#     
#     sel.features[[l]] <- probing(delta,m.gam$model[indices.abnormal[indices],-1],r=risk)
#     
#     #formula.NN <- as.formula(paste("delta~",paste(colnames(m.gam$model[,-1]),collapse="+")))
#     formula.NN <- as.formula(paste("delta~",paste(sel.features[[l]]$selected,collapse="+"),sep=""))
#     list.AWPF[[l]] <- neuralnet(formula.NN,data=abnormal.tot[indices,sel.features[[l]]$selected],hidden=4)
#     
#     
#   }
#   
#   to.return <- list("m.cluster"=m.cluster,"radii"=radii,"radii.mean"=radii.mean,
#     "list.AWPF"=list.AWPF,"sel.features"=sel.features,"list.delta"=list.delta,"abnormal"=abnormal.tot)
# }


##############################

names.for.probing <- c(paste("wp.p",1:12,sep=""),"ws.12",paste("ws.12.p",1:12,sep=""),paste("ws.12.n",1:11,sep=""),
      "PVs","PVp","mu.s","mu.p","RU.s.12","RD.s.12","RU.p","RD.p",
      paste("ws.angle.p",1:3,sep="",".12"),paste("ws.angle.n",1:3,sep="",".12"),"ws.angle.12",
      paste("ws.angle.p",1:3,sep="",".24"),paste("ws.angle.n",1:3,sep="",".24"),"ws.angle.24")

gam.data <- rbind(learning.new,valid.new)
gam.data <- gam.data[,-which(colnames(learning.new) %in% c("date","horizon.int"))]

##############################""
## FOR EVALUATION ##
# !!!!!!!!!!!!!!! ##
####################

init.date <- ymd_h(2010111812)
init.index.2 <- which(eval.new$date == init.date)

##########################"

learning.new <- learning.new[,-which(colnames(learning.new) %in% c("date","horizon.int","wp"))]
valid.new <- valid.new[,-which(colnames(valid.new) %in% c("date","horizon.int","wp"))]
eval.new <- eval.new[,-which(colnames(eval.new) %in% c("date","horizon.int","wp"))]

learning.new <- learning.new[,names.for.probing]
valid.new <- valid.new[,names.for.probing]
eval.new <- eval.new[,names.for.probing]

#####################################################"

delta.tot <- y.hat.lv - y

gam.formula <- as.formula(paste("delta.tot~",paste("s(",colnames(learning.new),")",collapse="+"),sep=""))
m.gam <- gam(gam.formula,data=gam.data)

# clusters = cluster.generation(y.hat=y.hat.lv,
#       y = y,
#       data=rbind(learning.new,valid.new)[,names.for.probing],
#      4,threshold=0.1,m.gam=m.gam,alpha=0.3,risk=0.15)

##############################

online.estimation <- function(n_ahead,data,data.corr,cluster.gen.output,n_clusters)
{
  y.hat <- rep(0,n_ahead)
  y.no.corr <- rep(0,n_ahead)
  
  X.online <- data[1:4,]
  
  ########################################
  
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
    y.no.corr[t] <- mean(as.numeric(y.t1))
    
    eps.c <- correction.engine(data.corr[t,],cluster.gen.output,n_clusters)
    
    y.hat[t] <- y.hat[t] - eps.c
    
    X.online <- rbind(X.online,data[tt1,])
    
    for(j in 2:4)
    {
      X.online[tt1,paste("wp_hn0",j,sep="")] <- X.online[tt,paste("wp_hn0",j-1,sep="")]
    }
    
    X.online$wp_hn01[tt1] <- y.hat[t]
    
    
  }
  
  to.return <- list("y.hat"=y.hat,"y.no.corr"=y.no.corr,"X.online"=X.online)
  
}
################################################

data = rbind(learning.new,valid.new)

y.hat.lv <- by(predict(lm.ensemble,newdata=rbind(lm.data.learn,lm.data.valid)),as.factor(rbind(learning.1,valid.1)$date),mean )
y <- by(rbind(lm.data.learn,lm.data.valid)$wp,as.factor(rbind(learning.1,valid.1)$date),mean)

data$delta <- y.hat.lv - y

indices.ab <- which(abs(data$delta) > 0.09)
n_clusters = 4
alpha = 0.5
name.response = "delta"
n_hidden = 4
r <- 0.1

cluster.info = cluster.generation(data[indices.ab,],indices.ab,n_clusters,alpha,m.gam,name.response,n_hidden,risk=r)


#################################################

n_ahead = 48

init.index <- which(eval.1$date == init.date)[1]

data = eval.1[init.index+(1:(4*n_ahead))-1,]
data.corr = eval.new[init.index.2+(1:n_ahead)-1,]

y = by(data$wp,as.factor(data$date),mean)

test1 = online.estimation(48,data,data.corr,cluster.info,n_clusters)

################################################
### OADMM ####
##############

learning.OADMM <- train[which(ymd_h(train$date) %in% dates.L),]
valid.OADMM <- train[which(ymd_h(train$date) %in% dates.V),]
eval.OADMM <- train[which(ymd_h(train$date) %in% dates.E),]

OADMM.data <- OADMM.prep(eval.OADMM[,-1],l=2,1,1:7)
OADMM.est <- OADMM(OADMM.data$X,OADMM.data$y,0.2,1,2,sigmoid.coeffs=c(0,1,1),maxit=nrow(OADMM.data$X))

plot(y,type="l")
#lines(test1$y.hat,col="red")
lines(test1$y.no.corr,col="blue")
lines(OADMM.est$y.chap[(init.index.2:(init.index.2+47))-2],col="red")
