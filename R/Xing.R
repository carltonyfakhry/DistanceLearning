#' This function computes Xing's global distance metric learning classification
#' algorithm.
#'
#' @description This function computes Xing's global distance metric learning
#'              classification algorithm as described in [1]. See the Vignette
#'              by using the command \code{browseVignette("DistanceLearning")}
#'              for an introduction to using Xing's global
#'              distance metric learning method.
#'
#' @usage XingMethod(Y, X, S = NULL, D = NULL,
#'                    learning_rate = 0.1, epsilon = 0.01,
#'                    error = 1e-10, max_iterations = 1000)
#'
#'
#' @param Y vector of non-negative integer labels corresponding to each data point.
#' @param X Input numeric matrix where each row is a data point whose
#'          label is the corresponding entry in \code{Y} and each column is a
#'          variable.
#' @param S A \code{n * 2} similarity matrix describing the constraints of data points with
#'              with the same class label.
#'              Each row of the matrix is a pair of indices of two data points in \code{X}
#'              which belong to the same class. For example, pair(1, 3)
#'              says that the first data point is in the same class as the third
#'              data point. Default value is \code{S = NULL} in which case
#'              \code{S} is computed in full. Use this parameter to define a smaller
#'              similarity matrix which is appropriate to your given problem e.g via sampling
#'              methods. The indices in \code{S} should range between 1 and \code{nrow(X)}.
#' @param D A \code{n * 2} disimilarity matrix describing the constraints of data points with
#'              with a different class label.
#'              Each row of the matrix is a pair of indices of two data points in \code{X}
#'              which belong to different classes. For example, pair(1, 3)
#'              says that the first data point is in a different class than the third
#'              data point. Default value is \code{D = NULL} in which case
#'              \code{D} is computed in full. Use this parameter to define a smaller disimilarity
#'              matrix which is appropriate to your given problem e.g via sampling
#'              methods. The indices in \code{D} should range between 1 and \code{nrow(X)}.
#' @param learning_rate The learning rate to be used in the solver. Default value is
#'                      is \code{learning_rate = 0.1}.
#' @param epsilon Threshold for convergence of the gradient method. Default value is
#'                \code{epsilon = 0.01}.
#' @param error Threshold to be used when projecting onto the constraint
#'              set. Default value is \code{error = 1e-10}.
#' @param max_iterations The maximum number of iterations to be processed
#'                       in the solver. Default value is \code{max_iterations = 1000}.
#'
#' @return This function returns a list with the following items:
#' \item{XingTransform}{The matrix under which the data was transformed. The
#'               multiplication of this matrix with its transpose gives
#'               the matrix used in the Mahalanobis
#'               metric.}
#' \item{TransformedX}{The transformed original data \code{X} which was transformed
#'                     using the Xing Transform i.e \eqn{TransformedX = X *
#'                     XingTransform}}.
#'
#' @details See the Vignette by using the command
#'          \code{browseVignette("DistanceLearning")}
#'          for an introduction to using Xing's method.
#'
#' @author Carl Tony Fakhry, Ping Chen, Rahul Kulkarni and Kourosh Zarringhalam
#'
#' @references [1] Eric P. Xing, Michael I. Jordan, Stuart J Russell, and
#'                 Andrew Y. Ng. Distance Metric Learning with Application
#'                 to Clustering with Side-Information. In S. Becker,
#'                 S. Thrun, and K. Obermayer, editors, Advances in Neural
#'                 Information Processing Systems 15, pages 521-528.
#'                 MIT Press, 2003.
#'
#' @examples # Load data from package DistanceLearning
#' library(DistanceLearning)
#' library(class)
#' fname <- system.file("extdata", "example_data.csv", package="DistanceLearning")
#' df <- read.csv(fname)
#' Y <- as.integer(df$y)
#' X <- as.matrix(df[,c(2,3)])
#' sample_points <- sample(1:nrow(X), 180, replace = FALSE)
#' subX <- X[sample_points,]
#' subY <- Y[sample_points]
#'
#' # Learn the metric, and get the transformed data
#' result <- XingMethod(subY, subX)
#' XingMetric <- result$XingTransform
#' transformedX <- result$transformedX
#'
#' # Get the accuracy of KNN classification without applying the new metric
#' yhat <- knn(subX, X[-sample_points,], subY, k = 5)
#' Accuracy <- length(which(Y[-sample_points] == yhat))/length(Y[-sample_points])
#' Accuracy
#'
#' # Get the accuracy of KNN classification after applying the new metric
#' transformednewX <- X[-sample_points,] %*% XingMetric
#' yhat2 <- knn(transformedX, transformednewX, subY, k = 5)
#' Accuracy2 <- length(which(Y[-sample_points] == yhat2))/length(Y[-sample_points])
#' Accuracy2
#'
#' @export
XingMethod <- function(Y, X, S = NULL, D = NULL,
                        learning_rate = 0.1, epsilon = 0.01,
                        error = 1e-10, max_iterations = 1000){


  # epsilon must be a positive numeric number and less than 1
  if(!is.numeric(epsilon) | length(epsilon) != 1 | epsilon <= 0 | epsilon >= 1){
    stop("epsilon parameter must be a positive numeric number and less than 1!")
  }

  # epsilon must be a positive numeric number and less than 1
  if(!is.numeric(error) | length(error) != 1 | error <= 0 | error >= 1){
    stop("error parameter must be a positive numeric number and less than 1!")
  }

  # learning_rate must be a positive numeric number and less than 1
  if(!is.numeric(learning_rate) | length(learning_rate) != 1 | learning_rate <= 0 | learning_rate >= 1){
    stop("learning_rate parameter must be a positive numeric number and less than 1!")
  }

  # max_iterations must be a positive integer
  if(!is.numeric(max_iterations) | length(max_iterations) != 1 | floor(max_iterations) != max_iterations | max_iterations <= 0){
    stop("max_iterations parameter must be a positive integer!")
  }

  # Check the inputs X and Y
  labels <- sort(unique(Y))
  check_YAndX(Y, X, labels, check_pairs = F)

  # Handle similarity matrix
  X_S <- NULL
  if(!is.matrix(S) & length(S) == 0 & is.null(S)){
    X_S <- getX_S(Y, X, labels)
  }else if(!is.matrix(S) | !is.integer(S) | ncol(S) != 2 | nrow(S) < 1 | max(S[,1]) > nrow(X) | max(S[,2]) > nrow(X) | min(S[,1]) < 1 | min(S[,2]) < 1 | any(S != floor(S))){
    stop("S should be a non-empty integer matrix with ncol(S) = 2, at least 1 row, and the elements in S should be >= 1 and <= nrow(X)!")
  }else{
    X_S <- sumOuterProducts(X, S, ncol(X))
  }

  # Handle difference matrix
  if(!is.matrix(D) & length(D) == 0 & is.null(D)){
    D <-  getD(Y, X, labels)
  }else if(!is.matrix(D) | !is.integer(D) | ncol(D) != 2 | nrow(D) < 1 | max(D[,1]) > nrow(X) | max(D[,2]) > nrow(X) | min(D[,1]) < 1 | min(D[,2]) < 1 | any(D != floor(D))){
    stop("D should be a non-empty integer matrix with ncol(D) = 2, at least 1 row, and the elements in D should be >= 1 and <= nrow(X)!")
  }

  weight_matrix <- X_S

  results <- xing(Y, X, labels, X_S, D, weight_matrix, learning_rate, error, epsilon, max_iterations)
  return(results)

}
