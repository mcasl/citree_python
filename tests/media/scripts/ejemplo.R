set.seed(42)

source("scripts/citree.r")
source("scripts/kcitree.r")
source("scripts/klda.r")
Sigma1 <- matrix(c(10,0, 0, 0, 2, 0, 0, 0, 1),3,3)
Sigma2 <- matrix(c(1,0, 0, 0, 10, 0, 0, 0, 2),3,3)
Sigma3 <- matrix(c(2,0, 0, 0, 1, 0, 0, 0, 10),3,3)

mu1 <- c(-10, 10, 0)
mu2 <- c(10, 10, 0)
mu3 <- c(0, 0, 10)

library(MASS)
blob1 <- mvrnorm(n = 10000, mu=mu1, Sigma=Sigma1)
blob2 <- mvrnorm(n = 10000, mu=mu2, Sigma=Sigma2)
blob3 <- mvrnorm(n = 10000, mu=mu3, Sigma=Sigma3)

X <- rbind(blob1, blob2, blob3)
indicesDesordenados <- sample(nrow(X))
X <- X[indicesDesordenados , ]
plot(X)

X.kcitree <- kcitree(data = X, ncenters = 20, iter.max.kmeans = 100, g.inv = FALSE)
X.klda <- klda(kcitree = X.kcitree, data = X, k = 3)
X.lda <- predict(X.klda$lda,newdata=X)
Z <- cbind(X, X.kcitree$kmeans$cluster, X.lda$class)
write.csv(Z, "datos.csv")
