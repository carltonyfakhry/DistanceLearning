#' This function computes the Neighborhood Component Analysis (NCA)
#' classification algorithm.
#'
#' @description This function computes the Neighborhood Component Analysis (NCA)
#'              classification algorithm as described in [1]. See the Vignette
#'              by using the command \code{browseVignette("DistanceLearning")}
#'              for an introduction to using NCA.
#'
#' @usage NCA(Y, X, max_iterations = 100, learning_rate = 0.01)
#'
#' @param Y vector of non-negative integer labels corresponding to each data point.
#' @param X Input numeric matrix where each row is a data point whose
#'          label is the corresponding entry in \code{Y} and each column is a
#'          variable.
#' @param max_iterations The maximum number of iterations to use in the solver
#'                   when learning the transformation matrix. The default
#'                   value is set to \code{max_iterations = 100}.
#' @param learning_rate The learning rate to be used in the solver when
#'                      learning the transformation matrix. The default
#'                      value is set to \code{learning_rate = 0.01}.
#'
#' @return This function returns a list with the following items:
#' \item{NCATransform}{The transformation matrix.}
#' \item{TransformedX}{The transformed original data \code{X} which was transformed
#'                     using the NCA trasformation matrix i.e \eqn{TransformedX = X *
#'                     NCATransform}}.
#'
#' @details See the Vignette by using the command
#'          \code{browseVignette("DistanceLearning")}
#'          for an introduction to using NCA.
#'
#' @author Carl Tony Fakhry, Ping Chen, Rahul Kulkarni and Kourosh Zarringhalam
#'
#' @references [1] J. Goldberger, S. Roweis, G. Hinton, and R.
#'                 Salakhutdinov. Neighbourhood components analysis.
#'                 In L. K. Saul, Y. Weiss, and L. Bottou, editors,
#'                 Advances in Neural Information Processing
#'                 Systems 17, pages 513-520, Cambridge, MA, 2005. MIT Press.
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
#' result <- NCA(subY, subX)
#' NCAMetric <- result$NCATransform
#' transformedX <- result$transformedX
#'
#' # Get the accuracy of KNN classification without applying the new metric
#' yhat <- knn(subX, X[-sample_points,], subY, k = 5)
#' Accuracy <- length(which(Y[-sample_points] == yhat))/length(Y[-sample_points])
#' Accuracy
#'
#' # Get the accuracy of KNN classification after applying the new metric
#' transformednewX <- X[-sample_points,] %*% NCAMetric
#' yhat2 <- knn(transformedX, transformednewX, subY, k = 5)
#' Accuracy2 <- length(which(Y[-sample_points] == yhat2))/length(Y[-sample_points])
#' Accuracy2
#'
#' @export
NCA <- function(Y, X, max_iterations = 100, learning_rate = 0.01){

  # learning_rate must be a positive number between 0 and 1
  if(!is.numeric(learning_rate) | length(learning_rate) != 1 | learning_rate <= 0 | learning_rate >= 1){
    stop("learning_rate must be a numeric number between 0 and 1!")
  }

  # max_iterations must be a positive integer
  if(length(max_iterations) != 1 | floor(max_iterations) != max_iterations | max_iterations <= 0){
    stop("max_iterations must be a positive integer!")
  }

  # Check the inputs X and Y
  labels <- sort(unique(Y))
  check_YAndX(Y, X, labels, check_pairs = TRUE)

  results <- nca(Y, X, labels, max_iterations, learning_rate)
  return(results)

}
