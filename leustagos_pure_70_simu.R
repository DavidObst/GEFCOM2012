rm(list=objects())
load("data/training.RData")
source("fn.base.R")
library(lubridate)
library(Utility)
library(gbm)
library(dplyr)

cols.common.pred <- c("date", "farm", "dist", "wp", "is_key", "is_training")
data.tr <- rbind(data.tr.other, data.test.other, data.tr.key)
data.test <- rbind(data.test.key)
rm(data.tr.other, data.test.other, data.tr.key, data.test.key)

#tr.farm.params <- expand.grid(farm = 1:7, dist = levels(data.tr$dist_half))
#tr.farm.params <- tr.farm.params[order(tr.farm.params$farm),]
#rownames(tr.farm.params) <- 1:nrow(tr.farm.params)
#r <- 1 # for test purposes
#farm <- 1 # for test purposes
#k <- 1 # for test purposes

# packages <- c("gbm","cvTools","Metrics","data.table", "foreach")
# fn.libraries(packages)

data.tr$dist.quarter = floor(as.numeric(as.character(data.tr$dist))/12) + 1

## Remove unwilling instants ##
data.tr$date = ymd_hms(data.tr$date)
to.remove <- which(data.tr$date > ymd_hms(20101231230000))
data.tr <- data.tr[-to.remove,]

## ws2 and ws3 ##

data.tr$ws2 = data.tr$ws^2
data.tr$ws3 = data.tr$ws^3


## Keep only instants with NWP horizon 36 to 48 hrs
data.tr <- data.tr[(data.tr$dist.quarter == 4),]
data.tr = arrange(data.tr,date)

## Def of learning, valid and eval sets ##
indices.L <- 1:floor(0.6*nrow(data.tr))
indices.V <- floor(0.6*nrow(data.tr)+1):floor(0.8*nrow(data.tr))
indices.E <- floor(0.8*nrow(data.tr)+1):nrow(data.tr)

learning <- data.tr[indices.L,]
validation <- data.tr[indices.V,]
evaluation <- data.tr[indices.E,]

B = 20 

error_df_sep = data.frame("MAE"=rep(0,B*7),"RMSE"=rep(0,B*7),"SDE"=rep(0,B*7))
error_df_com = data.frame("MAE"=rep(0,B),"RMSE"=rep(0,B),"SDE"=rep(0,B))

sde <- function(y,y_hat)
{
  eps = y - y_hat
  
  R = sqrt(mean((eps-mean(eps))^2))
  
  return(R)
}

NN <- floor(nrow(evaluation)/7)

####### Gradient Boosting model for farm 1 #########

for(b in 1:B)
{
  model.tr.gbm.1 <-
    gbm(
      formula =  wp ~ 
        ws + 
        ws.angle + 
        ws.angle.p1 + ws.angle.p2 + ws.angle.p3 + 
        #ws.angle.n1 + ws.angle.n2 + ws.angle.n3 +
        hour + month + year +
        dist,
      #set_seq_cut,
      data = learning,
      n.trees = 2000,
      interaction.depth = 12,
      n.minobsinnode = 30,
      shrinkage =  0.01,
      distribution = "gaussian",
      train.fraction = 1
    )
  
  
  ### Boosting on historic features ###
  model.tr.gbm.2 <- gbm(
    formula =  wp ~ 
      wp_hn01 + wp_hn02 + wp_hn03 + wp_hn04 + 
      dist + set_seq_cut + hour + month + year + 
      clust.farm + clust + begin + clust.pos,
    data = learning,
    n.trees = 1500,
    interaction.depth = 8,
    n.minobsinnode = 30,
    shrinkage =  0.03,
    distribution = "gaussian",
    train.fraction = 1
  )
  
  ###
  
  model.tr.gbm.3 <-
    gbm(
      formula =  wp ~ 
        ws + 
        ws.angle + 
        ws.angle.p1 + ws.angle.p2 + ws.angle.p3 + 
        #ws.angle.n1 + ws.angle.n2 + ws.angle.n3 +
        hour + month + year + 
        dist + 
        set_seq_cut +
        wp_hn01,
      data = learning,
      n.trees = 2000,
      interaction.depth = 12,
      n.minobsinnode = 30,
      shrinkage =  0.01,
      distribution = "gaussian",
      train.fraction = 1
    )
  
  
  lm.learning <- data.frame("y"=learning$wp,
                            "y1"=predict(model.tr.gbm.1,newdata=learning,n.trees=2000),
                            "y2"=predict(model.tr.gbm.2,newdata=learning,n.trees=1500),
                            "y3"=predict(model.tr.gbm.3,newdata=learning,n.trees=2000)) 
  
  lm.model <- lm(y~y1+y2+y3,data=lm.learning)
  
  lm.eval <- data.frame("y"=evaluation$wp, 
                        "y1"=predict(model.tr.gbm.1,newdata=evaluation,n.trees=2000),
                        "y2"=predict(model.tr.gbm.2,newdata=evaluation,n.trees=1500),
                        "y3"=predict(model.tr.gbm.3,newdata=evaluation,n.trees=2000))
  
  y.hat = predict(lm.model,newdata=lm.eval)
  
  
  
  #############################################################
  ###################### Learning #############################
  #############################################################
  
  ### Models by farm
  
  fn.register.wk()
  models.by.farm <- foreach (farm = 1:7) %dopar% {
    
    library(gbm)
    library(dplyr)
    
    learning <- data.tr[intersect(indices.L,which(data.tr$farm==farm)) ,]
    
    models.farm = list()
    models.farm[[1]] <-
      gbm(
        formula =  wp ~ 
          ws + 
          ws.angle + 
          ws.angle.p1 + ws.angle.p2 + ws.angle.p3 + 
          #ws.angle.n1 + ws.angle.n2 + ws.angle.n3 +
          hour + month + year +
          dist,
        #set_seq_cut,
        data = learning,
        n.trees = 2000,
        interaction.depth = 12,
        n.minobsinnode = 30,
        shrinkage =  0.01,
        distribution = "gaussian",
        train.fraction = 1
      )
    
    
    ### Boosting on historic features ###
    models.farm[[2]] <- gbm(
      formula =  wp ~ 
        wp_hn01 + wp_hn02 + wp_hn03 + wp_hn04 + 
        dist + set_seq_cut + hour + month + year + 
        clust.farm + clust + begin + clust.pos,
      data = learning,
      n.trees = 1500,
      interaction.depth = 8,
      n.minobsinnode = 30,
      shrinkage =  0.03,
      distribution = "gaussian",
      train.fraction = 1
    )
    
    ###
    
    models.farm[[3]] <-
      gbm(
        formula =  wp ~ 
          ws + 
          ws.angle + 
          ws.angle.p1 + ws.angle.p2 + ws.angle.p3 + 
          #ws.angle.n1 + ws.angle.n2 + ws.angle.n3 +
          hour + month + year + 
          dist + 
          set_seq_cut +
          wp_hn01,
        data = learning,
        n.trees = 2000,
        interaction.depth = 12,
        n.minobsinnode = 30,
        shrinkage =  0.01,
        distribution = "gaussian",
        train.fraction = 1
      )
    
    models.farm
  }
  fn.kill.wk()
  
  
  ###############################################
  ### Trained by distance ####
  ############################
  
  models.by.dist = list()
  tr.dist.params <- paste("(",seq(33,45,3),",",seq(36,48,3),"]",sep="")
  
  fn.register.wk()
  models.by.dist <- foreach (p =1:length(tr.dist.params)) %dopar% {
    
    library(gbm)
    library(dplyr)
    
    models.dist <- list()
    
    learning = data.tr[intersect(indices.L,which(data.tr$dist_cut==tr.dist.params[p])),]
    
    models.dist[[1]] <-
      gbm(
        formula =  wp ~ 
          ws + wd_cut +   
          ws.angle + 
          ws.angle.p1 + ws.angle.p2 + ws.angle.p3 + 
          hour + month + 
          farm + 
          set_seq_cut,
        data = learning,
        n.trees = 3000,
        interaction.depth = 8,
        n.minobsinnode = 30,
        shrinkage =  0.01,
        distribution = "gaussian",
        train.fraction = 1
      )
    
    models.dist[[2]] <-
      gbm(
        formula =  wp ~ 
          wp_hn01 + wp_hn02 + wp_hn03 + wp_hn04 + 
          set_seq_cut + hour + month + year + dist + 
          clust.farm + clust + clust.pos + begin,
        data = learning,
        n.trees = 2500,
        interaction.depth = 8,
        n.minobsinnode = 30,
        shrinkage =  0.03,
        distribution = "gaussian",
        train.fraction = 1
      )
    
    models.dist[[3]] <-
      gbm(
        formula =  wp ~ 
          ws + wd_cut +
          ws.angle +        
          ws.angle.p1 + ws.angle.p2 + ws.angle.p3 +
          hour + month + 
          farm + 
          set_seq_cut +
          wp_hn01,
        data = learning,
        n.trees = 2500,
        interaction.depth = 5,
        n.minobsinnode = 30,
        shrinkage =  0.02,
        distribution = "gaussian",
        train.fraction = 1
      )
    
    models.dist
  }
  fn.kill.wk()
  
  ##### Models trained on the whole set directly #####
  
  models.global = list()
  learning = data.tr[indices.L,]
  
  fn.register.wk()
  models.global <- foreach (r=1) %dopar% {
    
    library(gbm)
    library(dplyr)
    
    models.glob <- list()
    
    models.glob[[1]] <-
      gbm(
        formula =  wp ~ 
          ws + wd_cut +   
          ws.angle + 
          ws.angle.p1 + ws.angle.p2 + ws.angle.p3 + 
          hour + month + 
          farm + 
          set_seq_cut,
        data = learning,
        n.trees = 3500,
        interaction.depth = 8,
        n.minobsinnode = 30,
        shrinkage =  0.01,
        distribution = "gaussian",
        train.fraction = 1
      )
    
    
    models.glob[[2]] <-
      gbm(
        formula =  wp ~ 
          ws + wd_cut +
          ws.angle +        
          ws.angle.p1 + ws.angle.p2 + ws.angle.p3 +
          hour + month + 
          farm + 
          set_seq_cut +
          wp_hn01,
        data = learning,
        n.trees = 3500,
        interaction.depth = 5,
        n.minobsinnode = 30,
        shrinkage =  0.02,
        distribution = "gaussian",
        train.fraction = 1
      )
    
    models.glob
  }
  fn.kill.wk()
  
  #####################################################
  ### Combining all the estimates ###
  ###################################
  
  pred.by.farm <- function(list.models,data.tr,list.trees) 
  {
    ##list.models: list of 7 lists of each 3 models (ie 3 models per farm)
    
    fn.register.wk()
    pred.df <- foreach (farm = 1:7, .combine=rbind) %do% {
      
      indices.L <- 1:floor(0.6*nrow(data.tr))
      indices.E <- floor(0.8*nrow(data.tr)+1):nrow(data.tr)
      
      learning <- data.tr[intersect(indices.L,which(data.tr$farm==farm)),]
      
      pred.one.farm <- foreach(r = 1:length(list.models[[farm]]),.combine=cbind) %do%
      {
        library(gbm)
        library(dplyr)
        
        predict(list.models[[farm]][[r]],newdata=learning,n.trees=list.trees[r])
      }
      
      pred.one.farm = as.data.frame(pred.one.farm)
      pred.one.farm$date = learning$date
      pred.one.farm$farm = farm
      
      pred.one.farm
      
    }
    
    fn.kill.wk()
    
    colnames(pred.df)[1:3] = c("y1","y2","y3")
    
    pred.df <- arrange(pred.df,date)
    
    ##########################################
    
    eval.df <- foreach (farm = 1:7, .combine=rbind) %do% {
      
      indices.E <- floor(0.8*nrow(data.tr)+1):nrow(data.tr)
      
      evaluation <- data.tr[intersect(indices.E,which(data.tr$farm==farm)),]
      
      pred.one.farm <- foreach(r = 1:length(list.models[[farm]]),.combine=cbind) %do%
      {
        library(gbm)
        library(dplyr)
        
        predict(list.models[[farm]][[r]],newdata=evaluation,n.trees=list.trees[r])
      }
      
      pred.one.farm = as.data.frame(pred.one.farm)
      pred.one.farm$date = evaluation$date
      pred.one.farm$farm = farm
      
      pred.one.farm
      
    }
    
    fn.kill.wk()
    
    colnames(eval.df)[1:3] = c("y1","y2","y3")
    
    eval.df <- arrange(eval.df,date)
    
    to.return <- list("learning.df"=pred.df,"eval.df"=eval.df)
    
    return(to.return)
  }
  
  test = pred.by.farm(models.by.farm,data.tr,c(2000,1500,2000))
  
  #### By dist ####
  
  pred.by.dist <- function(list.models,data.tr,list.trees)
  {
    # fn.register.wk()
    tr.dist.params <- paste("(",seq(33,45,3),",",seq(36,48,3),"]",sep="")
    
    pred.df <- foreach (p = 1:length(tr.dist.params), .combine=rbind) %do% {
      
      indices.L <- 1:floor(0.6*nrow(data.tr))
      indices.E <- floor(0.8*nrow(data.tr)+1):nrow(data.tr)
      
      learning <- data.tr[intersect(indices.L,which(data.tr$dist_cut==tr.dist.params[p])),]
      
      pred.one.dist <- foreach(r = 1:length(list.models[[p]]),.combine=cbind) %do%
      {
        library(gbm)
        library(dplyr)
        
        predict(list.models[[p]][[r]],newdata=learning,n.trees=list.trees[r])
      }
      
      pred.one.dist = as.data.frame(pred.one.dist)
      pred.one.dist$date = learning$date
      pred.one.dist$farm = learning$farm
      
      pred.one.dist
      
    }
    
    ## For eval set ##
    
    eval.df <- foreach (p = 1:length(tr.dist.params), .combine=rbind) %do% {
      
      indices.E <- floor(0.8*nrow(data.tr)+1):nrow(data.tr)
      
      evaluation <- data.tr[intersect(indices.E,which(data.tr$dist_cut==tr.dist.params[p])),]
      
      pred.one.dist <- foreach(r = 1:length(list.models[[p]]),.combine=cbind) %do%
      {
        library(gbm)
        library(dplyr)
        
        predict(list.models[[p]][[r]],newdata=evaluation,n.trees=list.trees[r])
      }
      
      pred.one.dist = as.data.frame(pred.one.dist)
      pred.one.dist$date = evaluation$date
      pred.one.dist$farm = evaluation$farm
      
      pred.one.dist
      
    }
    
    colnames(pred.df)[1:3] = c("y4","y5","y6")
    
    pred.df <- arrange(pred.df,date)
    
    colnames(eval.df)[1:3] = c("y4","y5","y6")
    
    eval.df <- arrange(eval.df,date)
    
    to.return = list("learning.df"=pred.df,"eval.df"=eval.df)
    return(to.return)
  }
  
  test2 = pred.by.dist(models.by.dist,data.tr,c(300,2500,2500))
  
  #### Global #####
  pred.glob <- function(list.models,data.tr,list.trees)
  {
    indices.L <- 1:floor(0.6*nrow(data.tr))
    learning <- data.tr[indices.L,]
    
    indices.E <- floor(0.8*nrow(data.tr)+1):nrow(data.tr)
    evaluation <- data.tr[indices.E,]
    
    pred.df <- foreach(r=1:length(list.models),.combine=cbind) %do%
    {
      library(gbm)
      library(dplyr)
      
      predict(list.models[[1]][[r]],newdata=learning,n.trees=list.trees[r])
    }
    
    ## For evaluation ###
    eval.df <- foreach(r=1:length(list.models),.combine=cbind) %do%
    {
      library(gbm)
      library(dplyr)
      
      predict(list.models[[1]][[r]],newdata=evaluation,n.trees=list.trees[r])
    }
    
    pred.df = as.data.frame(pred.df)
    
    pred.df$date = learning$date
    pred.df$farm = learning$farm
    
    pred.df <- arrange(pred.df,date)
    
    colnames(pred.df)[1:2] = c("y7","y8")
    
    eval.df = as.data.frame(eval.df)
    
    eval.df$date = evaluation$date
    eval.df$farm = evaluation$farm
    
    eval.df <- arrange(eval.df,date)
    
    colnames(eval.df)[1:2] = c("y7","y8")
    
    to.return = list("learning.df"=pred.df,"eval.df"=eval.df)
    
    return(to.return)
  }
  
  test3 = pred.glob(models.global,data.tr,c(3500,3500))
  
  ################################################
  
  lm.df <- cbind(test[[1]],test2[[1]][,1:3],test3[[1]][,1:2],learning$wp)
  lm.df.eval = cbind(test[[2]],test2[[2]][,1:3],test3[[2]][,1:2],evaluation$wp)
  
  colnames(lm.df)[which(colnames(lm.df)=="learning$wp")] = "wp"
  colnames(lm.df.eval)[which(colnames(lm.df.eval)=="evaluation$wp")] = "wp"
  
  
  formula.leust <- as.formula(paste(c("wp~farm",paste("y",1:8,sep="")),collapse="+"))
  lm.leustagos <- lm(formula.leust,data=lm.df)
  
  y.hat.list = list()
  
  for (k in 1:7)
  {
    bb = (b-1)*7 + k
    
    
    
    y.hat.list[[k]] = predict(lm.leustagos,newdata=lm.df.eval[which(lm.df.eval$farm==k),])
    error_df_sep[["MAE"]][bb] = mae(evaluation$wp[which(evaluation$farm==k)],y.hat.list[[k]])
    error_df_sep[["RMSE"]][bb] = rmse(evaluation$wp[which(evaluation$farm==k)],y.hat.list[[k]])
    error_df_sep[["SDE"]][bb] = sde(evaluation$wp[which(evaluation$farm==k)],y.hat.list[[k]])
  }
  
  y.hat = predict(lm.leustagos,newdata=lm.df.eval)
  eps = evaluation$wp - y.hat
  
  error_df_com[b,] = c((1/NN)*sum(abs(eps)),
              sqrt((1/NN)*sum((eps)^2)),
              sqrt((1/NN)*sum((eps-mean(eps))^2)))
}



## Predict for all farms ##

y.hat.list = list()

mae = rep(0,7)
mse = rep(0,7)
rmse = rep(0,7)


for (k in 1:7)
{
  y.hat.list[[k]] = predict(lm.leustagos,newdata=lm.df.eval[which(lm.df.eval$farm==k),])
  mae[k] = mae(evaluation$wp[which(evaluation$farm==k)],y.hat.list[[k]])
  rmse[k] = rmse(evaluation$wp[which(evaluation$farm==k)],y.hat.list[[k]])
  mse[k] = (1/nrow(evaluation))*norm(evaluation$wp[which(evaluation$farm==k)]-y.hat.list[[k]],type=22)
}


eps.mat = rep(0,7)
for(k in 1:7)
{
  y = evaluation$wp[which(evaluation$farm==k)]
  kk = 1:length(y)
  
  eps.mat[k] = mean((y-y.hat.list[[k]])^2) 
}