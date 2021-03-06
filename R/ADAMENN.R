#' This function computes the Locally Adaptive Metric Nearest Neighbor (ADAMENN)
#' classification algorithm.
#'
#' @description This function computes the Locally Adaptive Metric Nearest
#'              Neighbor (ADAMENN) classification algorithm. See the Vignette
#'              by using the command \code{browseVignette("DistanceLearning")}
#'              for an introduction to using ADAMENN.  Note: Normalize the data
#'              before usage as suggested by the authors in [1].
#'
#'
#' @usage ADAMENN(Y, X, newX, K = 3, K_0 = NULL, K_1 = 3, K_2 = NULL, L = NULL, c = 5)
#'
#' @param Y vector of non-negative integer labels corresponding to each data point.
#' @param X Input numeric matrix where each row is a data point whose
#'          label is the corresponding entry in \code{Y} and each column is a
#'          variable.
#' @param newX A numeric matrix where each row is a data point whose label
#'             should be predicted using ADAMENN classification.
#'             If no prediction is required, \code{newX} should be set
#'             to an empty matrix with the number of columns equal to the
#'             number of columns of \code{X}.
#' @param K The number of neighbors to be used for K-NN classification after
#'          the ADAMENN metric has been learned. Default value is set to
#'          \code{K = 3}.
#' @param K_0 The number of neighbors to be used in learning the local measure
#'            of feature relevance for each data point in \code{newX}.
#'            Default value is \code{K_0 = NULL}, in which case \code{K_0} will
#'            be set to \code{K_0 = max(floor(0.1*length(Y)), 20)}.
#' @param K_1 The number of neighbors to be used in estimating the posterior
#'            class probability of each point in \code{newX}. Default value
#'            is \code{K_1 = 3}.
#' @param K_2 The number of neighbors to be used in estimating the posterior
#'            class probability conditioned on the value of a feature.
#'            Default value is \code{K_2 = NULL}, in which case \code{K_2} will
#'            be set to \code{K_2 = max(floor(0.15*length(Y)), 2)}.
#'            K_2 must be greater than K_1.
#' @param L The number of neighbors of a data point in \code{newX} to be used
#'          along each feature. Default value is \code{L = NULL}, in which
#'          case \code{L} will be set to \code{L = floor(K_2/2)}.
#'          \code{L} must be smaller or equal to \code{K_2}.
#' @param c A non-negative tuning paramter which scales the ADAMENN metric.
#'          Default value is set to \code{c = 5}.
#'
#' @return This function returns the predicted class labels of data points
#'         in \code{newX}. If a data point in \code{newX} violates the assumptions
#'         of the classifier then the data point cannot be classified and is
#'         assigned a label of -1.
#'
#' @details See the Vignette by using the command
#'          \code{browseVignette("DistanceLearning")}
#'          for a mathematical introduction to ADAMENN.
#'
#' @author Carl Tony Fakhry, Ping Chen, Rahul Kulkarni and Kourosh Zarringhalam
#'
#' @references [1] C. Domeniconi, D. Gunopulos, Locally adaptive metric nearest-neighbor
#'             classification, IEEE Transactions on Pattern Analysis and Machine Intelligence
#'             24 (2002) 1281-1285.
#'
#' @examples # Load data from package DistanceLearning
#' library(DistanceLearning)
#' fname <- system.file("extdata", "example_data.csv", package="DistanceLearning")
#' df <- read.csv(fname)
#' Y <- as.integer(df$y)
#' X <- scale(as.matrix(df[,c(2,3)]))
#' sample_points <- sample(1:nrow(X), 40, replace = FALSE)
#' newX <- X[sample_points,]
#' subY <- Y[sample_points]
#'
#' # Predict class labels for newX
#' Yhat <- ADAMENN(Y, X, newX)
#'
#' # Get the accuracy
#' Accuracy <- length(which(Yhat == subY))/length(subY)
#'
#' @seealso \code{\link{iADAMENN}}
#'
#' @export
ADAMENN <- function(Y, X, newX, K = 3, K_0 = NULL, K_1 = 3, K_2 = NULL, L = NULL, c = 5){

  # c must be a non-negative numeric number
  if(!is.numeric(c) | length(c) != 1 | c < 0){
    stop("c must be a non-negative numeric number!")
  }

  # K_0 must be a positive integer
  if(length(K_0) == 0 & is.null(K_0)){
    K_0 <- max(floor(0.1*length(Y)), 20)
  } else if(length(K_0) != 1 | floor(K_0) != K_0 | K_0 <= 0){
    stop("K_0 must be a positive integer. If K_0 is set to NULL then the default
         value K_0 = max(floor(0.1*length(Y)), 20) is used!")
  }

  # K_1 must be a positive integer
  if(length(K_1) != 1 | floor(K_1) != K_1 | K_1 <= 0){
    stop("K_1 must be a positive integer!")
  }

  # K_2 must be a positive integer
  if(length(K_2) == 0 & is.null(K_2)){
    K_2 <- max(floor(0.15*length(Y)), 2)
  } else if(length(K_2) != 1 | floor(K_2) != K_2 | K_2 <= 0 | K_2 <= K_1){
    stop("K_2 must be a positive integer greater than K_1. If K_2 is set to NULL then the default
         value K_2 = max(floor(0.15*length(Y)), 2) is used!")
  }

  # K must be a positive integer
  if(length(K) != 1 | floor(K) != K | K <= 0){
    stop("K must be a positive integer!")
  }

  # L must be a positive integer
  if(length(L) == 0 & is.null(L)){
    L <- floor(K_2/2)
  } else if(length(L) != 1 | floor(L) != L | L <= 0 | L > K_2){
    stop("L must be a positive integer less than or equal to K_2. If L is set to NULL then the default
         value L = floor(K_2/2) is used!")
  }

  # Check the inputs X and Y
  labels <- sort(unique(Y))
  check_YAndX(Y, X, labels, check_pairs = FALSE)
  check_newX(X, newX)

  n_iterations <- 1

  yhat <- adamenn(Y, X, newX, labels, K, K_0, K_1, K_2, L, c, n_iterations)
  return(yhat)

}



#' This function computes the iterative Locally Adaptive Metric Nearest
#' Neighbor (iADAMENN) classification algorithm.
#'
#' @description This function computes the iterative Locally Adaptive Metric
#'              Nearest Neighbor (iADAMENN) classification algorithm. See the
#'              Vignette by using the command
#'              \code{browseVignette("DistanceLearning")}
#'              for an introduction to using iADAMENN.
#'              Note: Normalize the data
#'              before usage as suggested by the authors in [1].
#'
#' @usage iADAMENN(Y, X, newX, K = 3, K_0 = NULL,
#'                        K_1 = 3, K_2 = NULL, L = NULL, c = 5, n_iterations = 5)
#'
#' @param Y vector of non-negative integer labels corresponding to each data point.
#' @param X Input numeric matrix where each row is a data point whose
#'          label is the corresponding entry in \code{Y} and each column is a
#'          variable.
#' @param newX A numeric matrix where each row is a data point whose label
#'             should be predicted using iADAMENN classification.
#'             If no prediction is required, \code{newX} should be set
#'             to an empty matrix with the number of columns equal to the
#'             number of columns of \code{X}.
#' @param K The number of neighbors to be used for K-NN classification after
#'          the iADAMENN metric has been learned. Default value is set to
#'          \code{K = 3}.
#' @param K_0 The number of neighbors to be used in learning the local measure
#'            of feature relevance for each data point in \code{newX}.
#'            Default value is \code{K_0 = NULL}, in which case \code{K_0} will
#'            be set to \code{K_0 = max(floor(0.1*length(Y)), 20)}.
#' @param K_1 The number of neighbors to be used in estimating the posterior
#'            class probability of each point in \code{newX}. Default value
#'            is \code{K_1 = 3}.
#' @param K_2 The number of neighbors to be used in estimating the posterior
#'            class probability conditioned on the value of a feature.
#'            Default value is \code{K_2 = NULL}, in which case \code{K_2} will
#'            be set to \code{K_2 = max(floor(0.15*length(Y)), 2)}.
#'            K_2 must be greater than K_1.
#' @param L The number of neighbors of a data point in \code{newX} to be used
#'          along each feature. Default value is \code{L = NULL}, in which
#'          case \code{L} will be set to \code{L = floor(K_2/2)}.
#'          \code{L} must be smaller or equal to \code{K_2}.
#' @param c A non-negative tuning paramter which scales the iADAMENN metric.
#'          Default value is set to \code{c = 5}.
#' @param n_iterations the number of iterations for which the iADAMENN metric
#'                     will be relearned.
#'
#' @return This function returns the predicted class labels of data points
#'         in \code{newX}. If a data point in \code{newX} violates the assumptions
#'         of the classifier then the data point cannot be classified and is
#'         assigned a label of -1.
#'
#' @details See the Vignette by using the command
#'          \code{browseVignette("DistanceLearning")}
#'          for a mathematical introduction to ADAMENN.
#'
#' @author Carl Tony Fakhry, Ping Chen, Rahul Kulkarni and Kourosh Zarringhalam
#'
#' @references [1] C. Domeniconi, D. Gunopulos, Locally adaptive metric nearest-neighbor
#'             classification, IEEE Transactions on Pattern Analysis and Machine Intelligence
#'             24 (2002) 1281-1285.
#'
#' @examples # Load data from package DistanceLearning
#' library(DistanceLearning)
#' fname <- system.file("extdata", "example_data.csv", package="DistanceLearning")
#' df <- read.csv(fname)
#' Y <- as.integer(df$y)
#' X <- scale(as.matrix(df[,c(2,3)]))
#' sample_points <- sample(1:nrow(X), 40, replace = FALSE)
#' newX <- X[sample_points,]
#' subY <- Y[sample_points]
#'
#' # Predict class labels for newX
#' Yhat <- iADAMENN(Y, X, newX)
#'
#' # Get the accuracy
#' Accuracy <- length(which(Yhat == subY))/length(subY)
#'
#' @seealso \code{\link{ADAMENN}}
#'
#' @export
iADAMENN <- function(Y, X, newX, K = 3, K_0 = NULL, K_1 = 3, K_2 = NULL, L = NULL, c = 5, n_iterations = 5){

  # n_iterations must be a positive integer
  if(length(n_iterations) != 1 | floor(n_iterations) != n_iterations | n_iterations < 1){
    stop("n_iterations must be a positive integer!")
  }

  # c must be a non-negative numeric number
  if(!is.numeric(c) | length(c) != 1 | c < 0){
    stop("c must be a non-negative numeric number!")
  }

  # K_0 must be a positive integer
  if(length(K_0) == 0 & is.null(K_0)){
    K_0 <- max(floor(0.1*length(Y)), 20)
  } else if(length(K_0) != 1 | floor(K_0) != K_0 | K_0 <= 0){
    stop("K_0 must be a positive integer. If K_0 is set to NULL then the default
         value K_0 = max(floor(0.1*length(Y)), 20) is used!")
  }

  # K_1 must be a positive integer
  if(length(K_1) != 1 | floor(K_1) != K_1 | K_1 <= 0){
    stop("K_1 must be a positive integer!")
  }

  # K_2 must be a positive integer
  if(length(K_2) == 0 & is.null(K_2)){
    K_2 <- max(floor(0.15*length(Y)), 2)
  } else if(length(K_2) != 1 | floor(K_2) != K_2 | K_2 <= 0 | K_2 <= K_1){
    stop("K_2 must be a positive integer greater than K_1. If K_2 is set to NULL then the default
         value K_2 = max(floor(0.15*length(Y)), 2) is used!")
  }

  # K must be a positive integer
  if(length(K) != 1 | floor(K) != K | K <= 0){
    stop("K must be a positive integer!")
  }

  # L must be a positive integer
  if(length(L) == 0 & is.null(L)){
    L <- floor(K_2/2)
  } else if(length(L) != 1 | floor(L) != L | L <= 0 | L > K_2){
    stop("L must be a positive integer less than or equal to K_2. If L is set to NULL then the default
         value L = floor(K_2/2) is used!")
  }

  # Check the inputs X, Y and newX
  labels <- sort(unique(Y))
  check_YAndX(Y, X, labels, check_pairs = FALSE)
  check_newX(X, newX)

  yhat <- adamenn(Y, X, newX, labels, K, K_0, K_1, K_2, L, c, n_iterations)
  return(yhat)

}
