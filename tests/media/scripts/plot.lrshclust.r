# ***********************************
# maptree drawclust convertida
# ***********************************
plot.lrshclust <- function (cluster, cex = par("cex"), srt=0, size = 2, col = NULL, labels=T, log.height=FALSE, border=0, xlab="", ylab="", main="", boxes=TRUE)
{
    if (class(cluster) == "lrshclust")
        clust <- cluster
    else stop("plot.lrshclust: input not lrshclust")
    if (log.height==TRUE) {clust$height <- log(clust$height)}
    par("srt"=srt)
    merg <- clust$merge
    nmerg <- nrow(merg)
    if (nmerg < 2)
        stop("draw: < 3 clusters")
    hite <- clust$height
    cord <- order(clust$order)
    xmax <- nrow(merg) + 1
    ymax <- max(hite)
    pinx <- par("pin")[1]
    piny <- par("pin")[2]
    xmin <- 1
    chr <- par("cin")[2] * cex
    box <- size * chr
    xscale <- (xmax - xmin)/pinx
    xbh <- xscale * box/2
    tail <- 0.25
    yscale <- ymax/(piny - tail)
    ytail <- yscale * tail
    ybx <- yscale * box
    ymin <- -ytail
    xf <- 0.1 * (xmax - xmin)
    yf <- 0.1 * (ymax - ymin)
    x1 <- xmin - xf
    x2 <- xmax + xf
    y1 <- ymin - yf
    y2 <- ymax + yf
    plot(c(x1, x2), c(y1, y2), type = "n", axes = FALSE, xlab = xlab,
        ylab = ylab, main=main)
    oldcex <- par("cex")
    par(cex = cex)
    if (is.null(col))
        kol <- rainbow(xmax)
    else if (length(col)==1) {
    	if (col == "gray" | col == "grey")
        	kol <- gray(seq(0.8, 0.2, length = xmax))
    } else kol <- col
    xmean <- rep(0, nmerg)
    i <- 1
    while (any(xmean == 0)) {
        if (xmean[i] == 0) {
            a <- merg[i, 1]
            b <- merg[i, 2]
            if (a < 0)
                x1 <- cord[-a]
            else x1 <- xmean[a]
            if (b < 0)
                x2 <- cord[-b]
            else x2 <- xmean[b]
            if (x1 != 0 && x2 != 0)
                xmean[i] <- mean(c(x1, x2))
        }
        i <- i + 1
        if (i > nmerg)
            i <- 1
    }
    for (i in 1:nmerg) {
        a <- merg[i, 1]
        b <- merg[i, 2]
        y2 <- hite[i]
        if (a > 0) {
            x1  <- xmean[a]
            y1a <- hite[a]
        }
        else {
            x1  <- cord[-a]
            y1a <- y2 - ytail
            if (boxes==TRUE) {
		px <- c(x1 - xbh, x1 + xbh, x1 + xbh, x1 - xbh, x1 - xbh)
		py <- c(y1a - ybx, y1a - ybx, y1a, y1a, y1a - ybx)
        	polygon(px, py, col = kol[-a], border = border)
		if (labels==TRUE) {
			text.default(x1, y1a - (ybx/2), as.character(-a))
		}  else if ( is.vector(labels) ) {
			text.default(x1, y1a - (ybx/2), labels[-a])
	    	}
	    }
        }
        if (b > 0) {
            x2 <- xmean[b]
            y1b <- hite[b]
        }
        else {
            x2 <- cord[-b]
            y1b <- y2 - ytail
            if (boxes==TRUE) {
            	px <- c(x2 - xbh, x2 + xbh, x2 + xbh, x2 - xbh, x2 - xbh)
            	py <- c(y1b - ybx, y1b - ybx, y1b, y1b, y1b - ybx)
		polygon(px, py, col = kol[-b], border = border)
		if (labels==TRUE) {
			text.default(x2, y1b - (ybx/2), as.character(-b))
		} else if ( is.vector(labels) ) {
			text.default(x2, y1b - (ybx/2), labels[-b])
            	}
	    }
        }
        lines(c(x1, x2), c(y2, y2))
        lines(c(x1, x1), c(y1a, y2))
        lines(c(x2, x2), c(y1b, y2))
    }
    par(cex = oldcex)
    invisible(kol)
}