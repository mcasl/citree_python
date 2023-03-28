require(MASS)
citree <- function(mu, v, nbin, Nbin, method="normal", g.inv=FALSE, keep.nodes=FALSE)
{
METHODS <- c("normal", "poisson", "multinomial")
    method <- pmatch(method, METHODS)
    if (is.na(method))
        stop("invalid clustering method")
    if (method == -1)
        stop("ambiguous clustering method")
    if (g.inv ) {inv <- ginv } else {inv <- solve }
#Initialization
	if (method==1) {         #normal
		ocupadas <- complete.cases(mu)
		mu <- mu[ocupadas,]
		v <- v[,,ocupadas]
		N <- nrow(mu)
                p <- ncol(mu)
		if (keep.nodes) {
	                mu.node <- array(0, dim=c(2*N-1,p))
	                v.node  <- array(0, dim=c(p,p,2*N-1) )
		        mu.node[1:N,] <- mu
	                v.node[,,1:N] <- v
		}
		v.inv <- array(, dim=dim(v))
 		for (i in 1:N ) {
			v.inv[,,i] <- inv(v[,,i])
		}
	} else if (method==2) {  #poisson
		ocupadas <- complete.cases(nbin)
		nbin <- as.vector(nbin)
		Nbin <- as.vector(Nbin)
		N <- length(nbin)
		ln.Nbin   <- log(Nbin)
		ln.nbin   <- log(nbin)
	} else if (method==3) {  #multinomial
		ocupadas <- complete.cases(nbin)
		N <- nrow(nbin)
 		p <- ncol(nbin)
 		Nbin <- apply(nbin, 1, sum)
 		ln.Nbin   <- log(Nbin)
 		ln.nbin   <- log(nbin)
 	}

        N.elementos <- N
        N.uniones <- N-1

 	height <- rep(0,N.uniones)
	agregacion <- -(1:N)
	uniones <- matrix(0, ncol=2, nrow=N.uniones)
	D <- double((N^2-N)/2)
	class(D)<- "dist"
	attr(D,"Size") <- N
	Dmin   <- Inf

  indice <- 1
#################################
#### Calculo de las distancias
	if (method==1) {         #normal
		indice <- 1
		for ( j in 1:(N-1) )	{
			for ( i in (j+1):N )	{
				u <- mu[j,,drop=F]-mu[i,,drop=F]
				D[indice] <- u %*% inv(v[,,j] + v[,,i]) %*% t(u)
				indice <- indice+1
 	  	}
 		}
  } else if (method==2) {  #poisson
			aux  <- nbin*(ln.nbin-ln.Nbin)
			aux2 <- outer(nbin,nbin,FUN="+")
			aux2 <- (aux2*(log(aux2)-log(outer(Nbin,Nbin,FUN="+"))))
			indice <- 1
			for ( j in 1:(N-1) )	{
				for ( i in (j+1):N )	{
					D[indice] <-aux[i]+aux[j]-aux2[i,j]
					D[indice] <- D[indice]+aux[i]+aux[j]-aux2[i,j]
					indice <- indice+1
 	  		}
 			}
	} else if (method==3) {  #multinomial
		aux  <- nbin*(ln.nbin-ln.Nbin)
		aux3 <- log(outer(Nbin,Nbin,FUN="+"))
		for (d in 1:p) {
			indice <- 1
			aux2 <- outer(nbin[,d],nbin[,d],FUN="+")
			aux2 <- aux2*(log(aux2)-aux3)
			for ( j in 1:(N-1) )	{
				for ( i in (j+1):N )	{
					D[indice] <- D[indice]+aux[i,d]+aux[j,d]-aux2[i,j]
					indice <- indice+1
 	  		}
 			}
	         }
 	}

  numiter <- N-2
  if (method==1) {        #normal
	  for(step in 1:numiter) {
			indice <- 1
			Dmin <- Inf
			j <- 0
			while (j < (N-1)) {
				j <- j+1
				i <- j+1
				while ( i <= N) {
					if (D[indice] < Dmin ) {
						Dmin <- D[indice]
						imin <- i
						jmin <- j
					}
					indice <- indice+1
					i <- i+1
			         }
	 		}
			uniones[step,1] <- agregacion[imin]
			uniones[step,2] <- agregacion[jmin]
			vmerged.inv <- v.inv[,,imin] + v.inv[,,jmin]
			vmerged <- inv(vmerged.inv)
                        xmerged <- vmerged %*% ( v.inv[,,imin] %*% t(mu[imin,,drop=F]) +  v.inv[,,jmin] %*% t(mu[jmin,,drop=F]))
			if (keep.nodes) {                 
			       mu.node[N.elementos+step,] <- xmerged
                        	v.node[,,N.elementos+step] <- vmerged
			}
			if ( imin==N ) {
				remplazo <- cbind(imin,(1:(imin-1))[-c(jmin)])
			} else if (imin ==2) {
				remplazo <- cbind((imin+1):N,imin)
			} else {
				remplazo <- rbind(cbind(imin,(1:(imin-1))[-c(jmin)]),cbind((imin+1):N,imin))
			}
			indices  <- ((remplazo[,2]-1)*(N-remplazo[,2]/2))+remplazo[,1]-remplazo[,2]
			indice	 <- 1
			for ( j in (1:N)[-c(imin, jmin)] )	{
				u <- t(xmerged) - mu[j,,drop=F]
				D[indices[indice]] <- u %*% inv( v[,,j] + vmerged ) %*% t(u)
				indice <- indice+1
			}
			mu[imin,] <- xmerged
			v[,,imin] <- vmerged
			v.inv[,,imin] <- vmerged.inv
			if (agregacion[imin] <0) { altura.1 <- 0} else {altura.1 <- height[agregacion[imin]]}
			if (agregacion[jmin] <0) { altura.2 <- 0} else {altura.2 <- height[agregacion[jmin]]}
		#prueba[step] <- c(altura.1, altura.2, Dmin, agregacion[imin],agregacion[jmin])
			height[step] <- altura.1 + altura.2 + Dmin
			agregacion[imin] <- step
			mu <- mu[ -c(jmin),]
			v  <-  v[,,-c(jmin)]
			v.inv <- v.inv[,,-c(jmin)]
			agregacion <- agregacion[-c(jmin)]
			if (jmin==1){
				indices <- cbind(2:N,1)
			}else {
				indices <- rbind( cbind(jmin, 1:(jmin-1)) ,  cbind((jmin+1):N,jmin) )
			}
	  	indices <- ((indices[,2]-1)*(N-indices[,2]/2))+indices[,1]-indices[,2]
	  	D <- D[-c(indices)]
	  	class(D) <- "dist"
	  	N <- N-1
	  	attr(D,"Size")<- N
		}
  } else if (method==2) { #poisson
	  for(step in 1:numiter) {
			indice <- 1
			Dmin <- Inf
			j <- 0
			while (j < (N-1)) {
				j <- j+1
				i <- j+1
				while ( i <= N) {
					if (D[indice] < Dmin ) {
						Dmin <- D[indice]
						imin <- i
				 		jmin <- j
				 	}
				 	indice <- indice+1
					i <- i+1
 	  		}
 			}
			uniones[step,1] <- agregacion[imin]
			uniones[step,2] <- agregacion[jmin]

			nbin.merged <- nbin[imin]+nbin[jmin]
			Nbin.merged <- Nbin[imin]+Nbin[jmin]
			ln.nbin.merged <- log(nbin.merged)
			ln.Nbin.merged <- log(Nbin.merged)
			if ( imin==N ) {
				remplazo <- cbind(imin,(1:(imin-1))[-c(jmin)])
			} else if (imin ==2) {
				remplazo <- cbind((imin+1):N,imin)
			} else {
				remplazo <- rbind(cbind(imin,(1:(imin-1))[-c(jmin)]),cbind((imin+1):N,imin))
			}
			indices  <- ((remplazo[,2]-1)*(N-remplazo[,2]/2))+remplazo[,1]-remplazo[,2]
			indice	 <- 1
			for ( j in (1:N)[-c(imin, jmin)] )	{
				D[indices[indice]] <- nbin.merged*(ln.nbin.merged-ln.Nbin.merged)+nbin[j]*(ln.nbin[j]-ln.Nbin[j])- (nbin.merged+nbin[j])*(log(nbin.merged+nbin[j])-log(Nbin.merged+Nbin[j]))
				indice <- indice+1
			}
			nbin[imin] <- nbin.merged
			Nbin[imin] <- Nbin.merged
			ln.nbin[imin] <- ln.nbin.merged
			ln.Nbin[imin] <- ln.Nbin.merged
			if (agregacion[imin] <0) { altura.1 <- 0} else {altura.1 <- height[agregacion[imin]]}
			if (agregacion[jmin] <0) { altura.2 <- 0} else {altura.2 <- height[agregacion[jmin]]}
			height[step] <- altura.1 + altura.2 + Dmin
			agregacion[imin] <- step
			nbin <- nbin[-c(jmin)]
			Nbin <- Nbin[-c(jmin)]
			ln.nbin <- ln.nbin[-c(jmin)]
			ln.Nbin <- ln.Nbin[-c(jmin)]

			agregacion <- agregacion[-c(jmin)]
			if (jmin==1){
				indices <- cbind(2:N,1)
			}else {
				indices <- rbind( cbind(jmin, 1:(jmin-1)) ,  cbind((jmin+1):N,jmin) )
			}
	  	indices <- ((indices[,2]-1)*(N-indices[,2]/2))+indices[,1]-indices[,2]
	  	D <- D[-c(indices)]
	  	class(D) <- "dist"
	  	N <- N-1
	  	attr(D,"Size")<- N
		}
	} else if (method==3) {  #multinomial
		for(step in 1:numiter) {
			indice <- 1
			Dmin <- Inf
			j <- 0
			while (j < (N-1)) {
				j <- j+1
				i <- j+1
				while ( i <= N) {
						if (D[indice] < Dmin ) {
							Dmin <- D[indice]
							imin <- i
				 			jmin <- j
				 		}
				 		indice <- indice+1
						i <- i+1
 	  		}
	 		}
			uniones[step,1] <- agregacion[imin]
			uniones[step,2] <- agregacion[jmin]
			nbin.merged <- nbin[imin,]+nbin[jmin,]
			Nbin.merged <- Nbin[imin]+Nbin[jmin]
			ln.nbin.merged <- log(nbin.merged)
			ln.Nbin.merged <- log(Nbin.merged)

			if ( imin==N ) {
				remplazo <- cbind(imin,(1:(imin-1))[-c(jmin)])
			} else if (imin ==2) {
				remplazo <- cbind((imin+1):N,imin)
			} else {
				remplazo <- rbind(cbind(imin,(1:(imin-1))[-c(jmin)]),cbind((imin+1):N,imin))
			}
			indices  <- ((remplazo[,2]-1)*(N-remplazo[,2]/2))+remplazo[,1]-remplazo[,2]

			D[indices]<-0
			for (d in 1:p) {
				indice	 <- 1
				for ( j in (1:N)[-c(imin, jmin)] )	{
					D[indices[indice]] <- D[indices[indice]] +	nbin.merged[d]*(ln.nbin.merged[d]-ln.Nbin.merged)+nbin[j,d]*(ln.nbin[j,d]-ln.Nbin[j])- (nbin.merged[d]+nbin[j,d])*(log(nbin.merged[d]+nbin[j,d])-log(Nbin.merged+Nbin[j]))
					indice <- indice+1
				}
			}

			nbin[imin,]    <- nbin.merged
			Nbin[imin]     <- Nbin.merged
			ln.nbin[imin,] <- ln.nbin.merged
			ln.Nbin[imin]  <- ln.Nbin.merged
			if (agregacion[imin] <0) { altura.1 <- 0} else {altura.1 <- height[agregacion[imin]]}
			if (agregacion[jmin] <0) { altura.2 <- 0} else {altura.2 <- height[agregacion[jmin]]}
			height[step] <- altura.1 + altura.2 + Dmin
			#browser()
			agregacion[imin] <- step

			nbin <- nbin[-c(jmin),]
			Nbin <- Nbin[-c(jmin)]
			ln.nbin <- ln.nbin[-c(jmin),]
			ln.Nbin <- ln.Nbin[-c(jmin)]

			agregacion <- agregacion[-c(jmin)]
			if (jmin==1){
				indices <- cbind(2:N,1)
			}else {
				indices <- rbind( cbind(jmin, 1:(jmin-1)) ,  cbind((jmin+1):N,jmin) )
			}
	  	indices <- ((indices[,2]-1)*(N-indices[,2]/2))+indices[,1]-indices[,2]
	  	D <- D[-c(indices)]
	  	class(D) <- "dist"
	  	N <- N-1
		  attr(D,"Size")<- N
		}
 	}

        imin <- 1
        jmin <- 2
        uniones[N.uniones, 1] <- agregacion[imin]
	uniones[N.uniones, 2] <- agregacion[jmin]
	if (agregacion[imin] <0) { altura.1 <- 0} else {altura.1 <- height[agregacion[imin]]}
	if (agregacion[jmin] <0) { altura.2 <- 0} else {altura.2 <- height[agregacion[jmin]]}
        height[N.uniones] <- altura.1 + altura.2 + D
	if (method==1) {         #normal
        	vmerged.inv <- v.inv[,,imin] + v.inv[,,jmin]
		vmerged <- inv(vmerged.inv)
	        xmerged <- vmerged %*% ( v.inv[,,imin] %*% t(mu[imin,,drop=F]) +  v.inv[,,jmin] %*% t(mu[jmin,,drop=F]))
		if (keep.nodes) {
		        mu.node[2*N.elementos-1,] <- xmerged
		        v.node[,,2*N.elementos-1] <- vmerged
		}
	}
#ordenacion al estilo de hcass2
	iorder <- matrix(0, ncol=N.uniones)
	iorder[1] <- uniones[N.uniones,1]
	iorder[2] <- uniones[N.uniones,2]
	LOC <- 2
	for(i in (N.uniones-1):1) {
  	for(j in 1:LOC) {
			if(iorder[j]==i) {
				iorder[j] <- uniones[i,1]
				if (j==LOC) {
                LOC <- LOC+1
                iorder[LOC] <- uniones[i,2]
       	}
       	else {
					LOC <- LOC+1
					for( k in LOC:(j+2) ) {
						iorder[k]=iorder[k-1]
					}
					iorder[j+1]=uniones[i,2]
				}
    	}
		}
	}
	buenas <- (1:length(ocupadas) )[ocupadas]
        result <- list(merge=uniones, height=height, order=-iorder, labels=buenas, method="Bin Aglomeration", dist.method="Likelihood Ratio Statistic", call = match.call())
	if (keep.nodes) {
	        result$mu.node <- mu.node
		result$V.node  <- v.node 

	}
	  class(result)<-  "lrshclust"
		return(result)
}


#tree.b <- citree(mu=dietas.b$mu, v=dietas.b$v)
