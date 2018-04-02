#' This function computes the Discriminant Adaptive Nearest Neighbor (DANN)
#' classification algorithm.
#'
#' @description This function computes the Discriminant Adaptive Nearest
#'              Neighbor (DANN) classification algorithm as described in [1].
#'              See the Vignette by using the command
#'              \code{browseVignette("DistanceLearning")}
#'              for an introduction to using DANN. Note: Normalize the data
#'              before usage as suggested by the authors in [1].
#'
#' @usage DANN(Y, X, newX, K_M = NULL, K = 5, epsilon = 1)
#'
#' @param Y vector of non-negative integer labels corresponding to each data point.
#' @param X Input numeric matrix where each row is a data point whose
#'          label is the corresponding entry in \code{Y} and each column is a
#'          variable.
#' @param newX A numeric matrix where each row is a data point whose label
#'             should be predicted using DANN classification.
#'             If no prediction is required, \code{newX} should be set
#'             to an empty matrix with the number of columns equal to the
#'             number of columns of \code{X}.
#' @param K_M The number of neighbors to be used in learning the DANN metric
#'            for each data point in \code{newX}.
#'            Default value is \code{K_M = NULL}, in which case \code{K_M} will
#'            be set to \code{K_M = max(50, floor(nrow(X)/5))} by default.
#' @param K The number of neighbors to be used for K-NN classification after
#'          the DANN metric has been learned. Default value is set to
#'          \code{K = 5}.
#' @param epsilon A tuning paramter which scales the DANN metric.
#'                Default value is set to \code{epsilon = 1}.
#'
#' @return This function returns the predicted class labels of data points
#'         in \code{newX}. If a data point in \code{newX} violates the assumptions
#'         of the classifier then the data point cannot be classified and is
#'         assigned a label of -1.
#'
#' @details See the Vignette by using the command
#'          \code{browseVignette("DistanceLearning")}
#'          for an introduction to using DANN.
#'
#' @author Carl Tony Fakhry, Ping Chen, Rahul Kulkarni and Kourosh Zarringhalam
#'
#' @references [1] T. Hastie, R. Tibshirani, Discriminant adaptive nearest neighbor
#'             classification, IEEE Transactions on Pattern Analysis and Machine
#'             Intelligence 18 (1996) 607-616.
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
#' Yhat <- DANN(Y[-sample_points], X[-sample_points,], newX)
#'
#' # Get the accuracy
#' Accuracy <- length(which(Yhat == subY))/length(subY)
#' Accuracy
#'
#' @seealso \code{\link{iDANN}}
#'
#' @export
DANN <- function(Y, X, newX, K_M = NULL, K = 5, epsilon = 1){

  # epsilon must be a positive numeric number
  if(!is.numeric(epsilon) | length(epsilon) != 1 | epsilon <= 0){
    stop("epsilon must be a non-negative numeric number!")
  }

  # K must be a positive integer
  if(length(K) != 1 | floor(K) != K | K <= 0){
    stop("K must be a positive integer!")
  }

  # K_M must be a positive integer if default is not used
  if(length(K_M) == 0 & is.null(K_M)){
    K_M <- max(50, floor(nrow(X)/5))
  } else if(length(K_M) != 1 | floor(K_M) != K_M | K_M <= 0){
    stop("K_M must be a positive integer. If K_M is set to NA then the default
         value K_M = max(50, floor(nrow(X)/5)) is used!")
  }

  # Check the inputs X, Y and newX
  labels <- sort(unique(Y))
  check_YAndX(Y, X, labels, check_pairs = FALSE)
  check_newX(X, newX)

  n_iterations <- 1
  Yhat <- dann(Y, X, newX, K, K_M, epsilon, n_iterations)
  return(Yhat)

}



#' This function computes the iterative Discriminant Adaptive Nearest Neighbor (DANN)
#' classification algorithm.
#'
#' @description This function computes the iterative Discriminant Adaptive Nearest
#'              Neighbor (iDANN) classification algorithm as described in [1].
#'              See the Vignette by using the command
#'              \code{browseVignette("DistanceLearning")}
#'              for an introduction to using iDANN.  Note: Normalize the data
#'              before usage as suggested by the authors in [1].
#'
#' @usage iDANN(Y, X, newX, K_M = NULL, K = 5, epsilon = 1, n_iterations = 5)
#'
#' @param Y vector of non-negative integer labels corresponding to each data point.
#' @param X Input numeric matrix where each row is a data point whose
#'          label is the corresponding entry in \code{Y} and each column is a
#'          variable.
#' @param newX A numeric matrix where each row is a data point whose label
#'             should be predicted using iDANN classification.
#'             If no prediction is required, \code{newX} should be set
#'             to an empty matrix with the number of columns equal to the
#'             number of columns of \code{X}.
#' @param K_M The number of neighbors to be used in learning the iDANN metric
#'            for each data point in \code{newX}.
#'            Default value is \code{K_M = NULL}, in which case \code{K_M} will
#'            be set to \code{K_M = max(50, floor(nrow(X)/5))} by default.
#' @param K The number of neighbors to be used for K-NN classification after
#'          the iDANN metric has been learned. Default value is set to
#'          \code{K = 5}.
#' @param n_iterations the number of iterations for which the DANN metric will
#'                     be relearned.
#' @param epsilon A tuning paramter which scales the iDANN metric.
#'                Default value is set to \code{epsilon = 1}.
#'
#' @return [1] This function returns the predicted class labels of data points
#'         in \code{newX}. If a data point in \code{newX} violates the assumptions
#'         of the classifier then the data point cannot be classified and is
#'         assigned a label of -1.
#'
#' @details See the Vignette by using the command
#'          \code{browseVignette("DistanceLearning")}
#'          for an introduction to using iDANN.
#'
#' @author Carl Tony Fakhry, Ping Chen, Rahul Kulkarni and Kourosh Zarringhalam
#'
#' @references T. Hastie, R. Tibshirani, Discriminant adaptive nearest neighbor
#'             classification, IEEE Transactions on Pattern Analysis and Machine
#'             Intelligence 18 (1996) 607-616.
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
#' Yhat <- iDANN(Y[-sample_points], X[-sample_points,], newX, n_iterations = 5)
#'
#' # Get the accuracy
#' Accuracy <- length(which(Yhat == subY))/length(subY)
#' Accuracy
#'
#' @seealso \code{\link{DANN}}
#'
#' @export
iDANN <- function(Y, X, newX, K_M = NULL, K = 5, epsilon = 1, n_iterations = 5){

  # n_iterations must be a positive integer
  if(length(n_iterations) != 1 | floor(n_iterations) != n_iterations | n_iterations < 1){
    stop("n_iterations must be a positive integer!")
  }

  # epsilon must be a non-negative numeric number
  if(!is.numeric(epsilon) | length(epsilon) != 1 | epsilon < 0){
    stop("epsilon must be a non-negative numeric number!")
  }

  # K must be a positive integer
  if(length(K) != 1 | floor(K) != K | K <= 0){
    stop("K must be a positive integer!")
  }

  # K_M must be a positive integer if default is not used
  if(length(K_M) == 0 & is.null(K_M)){
    K_M <- max(50, floor(nrow(X)/5))
  } else if(length(K_M) != 1 | floor(K_M) != K_M | K_M <= 0){
    stop("K_M must be a positive integer. If K_M is set to NULL then the default
         value K_M = max(50, floor(nrow(X)/5)) is used!")
  }

  # Check the inputs X and Y
  labels <- sort(unique(Y))
  check_YAndX(Y, X, labels, check_pairs = FALSE)
  check_newX(X, newX)

  Yhat <- dann(Y, X, newX, K, K_M, epsilon, n_iterations)
  return(Yhat)

}
