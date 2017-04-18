correction.engine.prep <- function(data,variables,name.response,n_hidden)
{
  formula <- paste(name.response,"~",paste(variables,collapse="+"),sep="")
  NN <- neuralnet::neuralnet(formula,data,hidden=n_hidden)
  
  y.hat <- as.numeric(neuralnet::compute(NN,data[,variables])$net.result)
  y <- data[,name.response]
  
  to.return <- list("NN"=NN,"y.hat"=y.hat,"y"=y)
  
  return(to.return)
}

#####################################################

cluster.generation <- function(data,indices.ab,n_clusters,alpha,m.gam,name.response,n_hidden,risk=0.1)
{
  m.clusters <- kcca(data,k=n_clusters)
  cluster.vector <- m.clusters@second
  centers <- m.clusters@centers
  
  radii <- rep(0,n_clusters)
  radii.mean <- rep(0,n_clusters)
  
  selected <- list()
  cluster.info <- list()
  cluster.NN <- list()
  
  
  
  for(l in 1:n_clusters)
  {
    indices <- which(cluster.vector==l)
    indices.ab.2 <- indices.ab[indices]
    
    distances <- apply(data[indices,],1,dist,y=centers[l,],type=2)
    radii[l] <- alpha*max(distances)
    radii.mean[l] <- mean(distances)
    
    sel = Probing::probing(m.gam$model[indices.ab.2,1],as.data.frame(m.gam$model[indices.ab.2,-1]),risk)$selected
    
    if(!is.null(sel))
    {
      selected[[l]] <- sel
      cluster.NN[[l]] <- correction.engine.prep(data[indices,],selected[[l]],name.response,n_hidden)
    }
    
    else{
      selected[[l]] <- "NULL"
      cluster.NN[[l]] <- "NULL"
    }
    
    
  }
  
  R <- list("centers"=centers,"m.clusters"=m.clusters,"radii"=radii,"radii.mean"=radii.mean,"selected"=selected,
        "cluster.NN"=cluster.NN)
  
  return(R)
  
}

######################################################

correction.engine <- function(datum,cluster.gen.output,n_clusters)
{
  eps <- 0
  
  for(l in 1:n_clusters)
  {
    if(!identical(cluster.gen.output$selected[[l]],"NULL"))
    {
      dist <- dist(datum,cluster.gen.output$centers[l,-59],type=2) ###To remove Delta... to edit later to generalize
      
      if(dist < cluster.gen.output$radii[l])
      {
        eps <- as.numeric(neuralnet::compute(cluster.gen.output$cluster.NN[[l]]$NN,
            datum[,cluster.gen.output$selected[[l]]])$net.result)
      }
    }
  
  }
  
  return(eps)
}

# complete.abnormal.engine <- function(y,y.hat,data,alpha,threshold,m.gam,n_clusters)
# {
#   indices.abnormal <- which(abs(y.hat - y) > threshold)
# 
#   ### Cluster generation ###
#   
#   cluster.info <- cluster.generation(data,indices.abnormal,n_clusters,alpha,m.gam,risk=0.1)
#   
#   ### Preparation of NN for each cluster ###
#   
#   for(l in 1:n_clusters)
#   {
#     
#   }
#   
#   correction.engine.prep <- function(data,variables,name.response,n_hidden)
#   
#   
# }