#require(MASS)
klda <- function(kcitree, data.kcitree, k) {
	result        <- list()
	result$clases <- cutree(kcitree$citree,k)[kcitree$kmeans$cluster]
	result$lda    <- lda(x=data.kcitree, grouping=result$clases)
	plot(result$lda, col=result$clases)
	return(result)
}


