#' This function computes the Relevant Component Analysis (RCA)
#' algorithm.
#'
#' @description This function computes the Relevant Component Analaysis
#'              (RCA) algorithm as desribed in [1][2]. See the Vignette
#'              by using the command \code{browseVignette("DistanceLearning")}
#'              for a mathematical introduction to using RCA.
#'
#' @usage RCA(X, chunklets, total_dims = NULL)
#'
#' @param X Input numeric matrix where each row is a data point whose
#'          label is the corresponding entry in \code{Y} and each column is a
#'          variable.
#' @param chunklets An integer vector indicating to which chunklet each data
#'                  point belongs to. If a data point does not belong to
#'                  a chunklet, then its chunklet value should be set to -1
#'                  otherwise all chunklets should be represented by
#'                  a non-negative integer. All data points in the same chunklet
#'                  must have the same class label.
#' @param total_dims The number of dimensions on which the data should be projected
#'                   on after the RCA metric has been learned. Default value is set
#'                   to \code{total_dims = NULL} in which case the value
#'                   \code{total_dims = ncol(X)} is used as default.
#'
#' @return This function returns a list with the following items:
#' \item{RCATransform}{The transformation matrix.}
#' \item{TransformedX}{The transformed original data \code{X} which was transformed
#'                     using the RCA trasformation matrix i.e \eqn{TransformedX = X *
#'                     RCATransform}}.
#'
#' @details See the Vignette by using the command
#'          \code{browseVignette("DistanceLearning")}
#'          for an introduction to using RCA.
#'
#' @author Carl Tony Fakhry, Ping Chen, Rahul Kulkarni and Kourosh Zarringhalam
#'
#' @references [1] Aharon Bar-Hillel, Tomer Hertz, Noam Shental, and Daphna
#'                 Weinshall. Learning distance functions using equivalence
#'                 relations. In In Proceedings of the Twentieth International
#'                 Conference on Machine Learning, pages 11-18.
#'
#'             [2] N. Shental, T. Hertz, D. Weinshall, and M. Pavel. Adjustment
#'                 learning and relevant component analysis. In Proceedings of
#'                 the Seventh European Conference on Computer Vision, volume
#'                 4, pages 776-792, Copenhagen, Denmark, 2002.
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
#' D <- matrix(c(0,1,1,0), nrow = 2, ncol = 2)
#'
#' # Predict class labels for newX and get the metric
#' result <- DCA(subX, chunklets = subY, D = D)
#' DCAMetric <- result$DCATransform
#' transformednewX <- result$transformedX
#'
#' # Get the accuracy of KNN classification without applying the new metric
#' yhat <- knn(subX, X[-sample_points,], subY, k = 5)
#' Accuracy <- length(which(Y[-sample_points] == yhat))/length(Y[-sample_points])
#' Accuracy
#'
#' # Get the accuracy of KNN classification after applying the new metric
#' transformedX <- X[-sample_points,] %*% DCAMetric
#' yhat2 <- knn(transformednewX, transformedX, subY, k = 5)
#' Accuracy2 <- length(which(Y[-sample_points] == yhat2))/length(Y[-sample_points])
#' Accuracy2
#'
#' @seealso \code{\link{DCA}}
#'
#' @export
RCA <- function(X, chunklets, total_dims = NULL){

  # Check the inputs X
  check_X_Chunklets(X, chunklets)

  # total_dims must be a positive integer if default is not used
  if(length(total_dims) == 0 & is.null(total_dims)){
    total_dims <- ncol(X)
  } else if(length(total_dims) != 1 | floor(total_dims) != total_dims | total_dims <= 0 | total_dims > ncol(X)){
    stop("total_dims must be a positive integer less than or equal to ncol(X) and greater or equal than 1!")
  }

  unique_chunklets <- unique(chunklets)

  results <- rca(X, chunklets, unique_chunklets, total_dims)
  return(results)

}
