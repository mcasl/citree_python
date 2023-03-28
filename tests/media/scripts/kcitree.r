#data es un array con los datos
#data.kmeans es el resultado del kmeans de data.



kcitree <- function(data, ncenters, iter.max.kmeans=40, g.inv=FALSE) {
   data.kmeans <- kmeans(data, ncenters, iter.max=iter.max.kmeans)
   p <- ncol(data.kmeans$centers)
   v<- array(0, dim=c(p,p,ncenters) )
   for (i in 1:ncenters ) {
      v[,,i] <- cov(data[data.kmeans$cluster==i,])/data.kmeans$size[i]
   }
   data.citree <- citree(mu=data.kmeans$centers, v=v, method="normal", g.inv=g.inv)
   return(list(citree=data.citree, kmeans=data.kmeans, mu=data.kmeans$centers, v=v))
}

#ejemplo de aplicacion
#arbol<- kcitree(data=food.data, ncenters=90, iter.max.kmeans=100)


