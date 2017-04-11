rm(list=objects())
library(magrittr)

lag.gen <- function(y,k=1)
{
  N <- length(y)
  R <- rep(0,N)
  
  if(k > 0)
  {
    R <- c(rep(0,k),y[1:(N-k)])
  }
  
  else if(k == 0) {R <- y}
  
  else{
    R <- c(y[(-k+1):N],rep(y[N],-k))
  }
  
  return(R)
}


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

#######################################################################

wf <- read.csv("data/train.csv",sep=",",dec=".",header=T)
#wf1 <- read.csv("../GEFCom2012-Wind-R/GEFCOM_neural_network/data/pre_treated/wf1.csv",sep=",",header=T,dec=".")
wf1 <- read.csv("data/windforecasts_wf1.csv",sep=",",dec=".",header=T)

wf$date <- ymd_h(wf$date)

wf1$issue <- ymd_h(wf1$date)
wf1$date <- ymd_h(wf1$date)+hours(wf1$hors)
wf1 <- wf1 %>% arrange(date)

to.drop <- which((wf1$date > ymd_h(2010123100)) | (wf1$date < ymd_h(2009070213)) )
to.drop.2 <- which((wf$date > ymd_h(2010123100)) | (wf$date < ymd_h(2009070213)))

wf <- wf[-to.drop.2,]
wf1 <- wf1[-to.drop,]

wf1["horizon.int"] <- sapply(wf1$hors,pred.cut.func,splitting=12)

wf1.new <- data.frame("date"=wf$date,"wp"=wf$wp1)

for(l in 1:12)
{
  wp.lagged <-  lag(wf1.new$wp,l)
  wp.lagged[which(is.na(wp.lagged))] <- 0
  wf1.new[,paste("wp",l,sep=".p")] <- wp.lagged
  
}

splitted <- split(wf1,as.factor(wf1$horizon.int))
names <- c("u","v","ws","wd")
names.2 <- c("u","v","ws")


col.names <- c()
col.names.2 <- c()


for(k in 1:length(names))
{
  col.names <- c(col.names,paste(names[k],c(12,24,36,48),sep="."))
  
}

for(k in 1:length(names.2))
{
  col.names.2 <- c(col.names.2,paste(names.2[k],c(12,24,36,48),sep="."))
}

for(k in 1:4)
{
  wf1.new[paste(names,k*12,sep=".")] <- splitted[[k]][names]#/sapply(splitted[[k]][names],max)
  wf1.new$issued <- splitted[[k]]$issued
  wf1.new$horizon.int <- splitted[[k]]$horizon.int
}

indices.normalize1 = which(colnames(wf1.new) %in% col.names.2)
normalize1 = apply(wf1.new[,indices.normalize1],2,max)

indices.normalize2 = which(colnames(wf1.new) %in% c("wd.12","wd.24","wd.36","wd.48"))

wf1.new[,indices.normalize1] = sweep(wf1.new[,indices.normalize1],2,normalize1,"/")
wf1.new[,indices.normalize2] = (wf1.new[,indices.normalize2] - 180)/180

for(l in setdiff(seq(-11,12,1),0))
{
  for(k in 1:length(col.names))
  {
    if(l > 0)
    {
      wf1.new[paste(col.names[k],l,sep=".p")] <- lag.gen(wf1.new[,col.names[k]],l)
    }
    else{
      wf1.new[paste(col.names[k],abs(l),sep=".n")] <- lag.gen(wf1.new[,col.names[k]],l)
    }
  }
}

#####################################
### Other features ####

names.ws <- c("ws.12","ws.24","ws.36","ws.48")

PVs <- apply(wf1.new[,c(paste(names.ws,1:11,sep=".n"),paste(names.ws,1:12,sep=".p"))],1,max) -
  apply(wf1.new[,c(paste(names.ws,1:11,sep=".n"),paste(names.ws,1:12,sep=".p"))],1,min)
wf1.new$PVs = PVs

mu.s = rowSums(wf1.new[,c(paste(names.ws,1:11,sep=".n"),paste(names.ws,1:12,sep=".p"))] )/24
wf1.new$mu.s = mu.s


## RU and RD ##

RU <- array(0,dim=c(nrow(wf1.new),23))

for(k in 1:4)
{
  for(t in 1:11)
  {
    RU[,t] <- wf1.new[,paste(paste("ws",12*k,sep="."),t,sep=".p")] - wf1.new[,paste(paste("ws",12*k,sep="."),t+1,sep=".p")]
  }
  
  for(t in 0:10)
  {
    if(t==0)
    {
      RU[,11+t+1] <- wf1.new[,paste("ws",12*k,sep=".")] - wf1.new[,paste(paste("ws",12*k,sep="."),".p1",sep="")]
    }
    
    else if(t==1)
    {
      RU[,11+t+1] <- wf1.new[,paste(paste("ws",12*k,sep="."),".n1",sep="")] -  wf1.new[,paste("ws",12*k,sep=".")] 
    }
    
    else
    {
      RU[,11+t+1] <- wf1.new[,paste(paste("ws",12*k,sep="."),t+1,sep=".n")] - wf1.new[,paste(paste("ws",12*k,sep="."),t,sep=".n")]
    }
    
    
  }
  
  RU.s <- apply(RU,1,max)
  RD.s <- apply(-RU,1,max)
  
  wf1.new[,paste("RU.s",12*k,sep=".")] <- RU.s
  wf1.new[,paste("RD.s",12*k,sep=".")] <- RD.s
  
}



######################################
N <- nrow(wf1.new)

indices.L <- 1:floor(0.6*N)
indices.V <- floor(0.6*N+1):floor(0.8*N)
indices.E <- floor(0.8*N+1):N

data.L <- wf1.new[indices.L,]

formula <- paste(colnames(data.L[,-which(colnames(data.L) %in% c("date","horizon.int","wp"))]),collapse="+")

formula <- as.formula(paste("wp~",formula,sep=""))

NN <- neuralnet(formula,data=data.L,hidden=4)
