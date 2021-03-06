% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ADAMENN.R
\name{iADAMENN}
\alias{iADAMENN}
\title{This function computes the iterative Locally Adaptive Metric Nearest
Neighbor (iADAMENN) classification algorithm.}
\usage{
iADAMENN(Y, X, newX, K = 3, K_0 = NULL,
                       K_1 = 3, K_2 = NULL, L = NULL, c = 5, n_iterations = 5)
}
\arguments{
\item{Y}{vector of non-negative integer labels corresponding to each data point.}

\item{X}{Input numeric matrix where each row is a data point whose
label is the corresponding entry in \code{Y} and each column is a
variable.}

\item{newX}{A numeric matrix where each row is a data point whose label
should be predicted using iADAMENN classification.
If no prediction is required, \code{newX} should be set
to an empty matrix with the number of columns equal to the
number of columns of \code{X}.}

\item{K}{The number of neighbors to be used for K-NN classification after
the iADAMENN metric has been learned. Default value is set to
\code{K = 3}.}

\item{K_0}{The number of neighbors to be used in learning the local measure
of feature relevance for each data point in \code{newX}.
Default value is \code{K_0 = NULL}, in which case \code{K_0} will
be set to \code{K_0 = max(floor(0.1*length(Y)), 20)}.}

\item{K_1}{The number of neighbors to be used in estimating the posterior
class probability of each point in \code{newX}. Default value
is \code{K_1 = 3}.}

\item{K_2}{The number of neighbors to be used in estimating the posterior
class probability conditioned on the value of a feature.
Default value is \code{K_2 = NULL}, in which case \code{K_2} will
be set to \code{K_2 = max(floor(0.15*length(Y)), 2)}.
K_2 must be greater than K_1.}

\item{L}{The number of neighbors of a data point in \code{newX} to be used
along each feature. Default value is \code{L = NULL}, in which
case \code{L} will be set to \code{L = floor(K_2/2)}.
\code{L} must be smaller or equal to \code{K_2}.}

\item{c}{A non-negative tuning paramter which scales the iADAMENN metric.
Default value is set to \code{c = 5}.}

\item{n_iterations}{the number of iterations for which the iADAMENN metric
will be relearned.}
}
\value{
This function returns the predicted class labels of data points
        in \code{newX}. If a data point in \code{newX} violates the assumptions
        of the classifier then the data point cannot be classified and is
        assigned a label of -1.
}
\description{
This function computes the iterative Locally Adaptive Metric
             Nearest Neighbor (iADAMENN) classification algorithm. See the
             Vignette by using the command
             \code{browseVignette("DistanceLearning")}
             for an introduction to using iADAMENN.
             Note: Normalize the data
             before usage as suggested by the authors in [1].
}
\details{
See the Vignette by using the command
         \code{browseVignette("DistanceLearning")}
         for a mathematical introduction to ADAMENN.
}
\examples{
# Load data from package DistanceLearning
library(DistanceLearning)
fname <- system.file("extdata", "example_data.csv", package="DistanceLearning")
df <- read.csv(fname)
Y <- as.integer(df$y)
X <- scale(as.matrix(df[,c(2,3)]))
sample_points <- sample(1:nrow(X), 40, replace = FALSE)
newX <- X[sample_points,]
subY <- Y[sample_points]

# Predict class labels for newX
Yhat <- iADAMENN(Y, X, newX)

# Get the accuracy
Accuracy <- length(which(Yhat == subY))/length(subY)

}
\references{
[1] C. Domeniconi, D. Gunopulos, Locally adaptive metric nearest-neighbor
            classification, IEEE Transactions on Pattern Analysis and Machine Intelligence
            24 (2002) 1281-1285.
}
\seealso{
\code{\link{ADAMENN}}
}
\author{
Carl Tony Fakhry, Ping Chen, Rahul Kulkarni and Kourosh Zarringhalam
}
