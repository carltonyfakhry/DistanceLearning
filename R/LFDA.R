#' This function computes the Linear Fisher Discriminant Analaysis (LFDA)
#' classification algorithm.
#'
#' @description This function computes the Linear Fisher Discriminant Analaysis
#'              (LFDA) classification algorithm as desribed in [1][2].
#'              See the Vignette by using the command
#'              \code{browseVignette("DistanceLearning")}
#'              for an introduction to using LFDA.
#'
#' @usage LFDA(Y, X, metric = "plain", total_dims = NULL)
#'
#' @param Y vector of non-negative integer labels corresponding to each data point.
#' @param X Input numeric matrix where each row is a data point whose
#'          label is the corresponding entry in \code{Y} and each column is a
#'          variable.
#' @param metric The possible options are: \code{plain}, \code{weighted} or
#'               \code{orthogonalized}. \code{metric = plain} is used as
#'               default.
#' @param total_dims The number of dimensions on which the data should be projected
#'                   on after the LFDA metric has been learned. Default value is set
#'                   to \code{total_dims = NULL} in which case the value
#'                   \code{total_dims = ncol(X)} is used as default.
#'
#' @return This function returns a list with the following items:
#' \item{LFDATransform}{The transformation matrix.}
#' \item{TransformedX}{The transformed original data \code{X} which was transformed
#'                     using the LFDA trasformation matrix i.e \eqn{TransformedX = X *
#'                     LFDATransform}}.
#'
#'
#' @details See the Vignette by using the command
#'          \code{browseVignette("DistanceLearning")}
#'          for an introduction to using LFDA.
#'
#' @author Carl Tony Fakhry, Ping Chen, Rahul Kulkarni and Kourosh Zarringhalam
#'
#' @references [1] Masashi Sugiyama. Dimensionality reduction of multimodal labeled data by
#'                 local Fisher discriminant analysis. Journal of Machine
#'                 Learning Research, vol.8, 1027--1061.
#'
#'             [2] Masashi Sugiyama. Local Fisher discriminant analysis for supervised
#'                 dimensionality reduction. In W. W. Cohen and A. Moore
#'                 (Eds.), Proceedings of 23rd International Conference on
#'                 Machine Learning (ICML2006), 905--912.
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
#' result <- LFDA(subY, subX)
#' LFDAMetric <- result$LFDATransform
#' transformedX <- result$transformedX
#'
#' # Get the accuracy of KNN classification without applying the new metric
#' yhat <- knn(subX, X[-sample_points,], subY, k = 5)
#' Accuracy <- length(which(Y[-sample_points] == yhat))/length(Y[-sample_points])
#' Accuracy
#'
#' # Get the accuracy of KNN classification after applying the new metric
#' transformednewX <- X[-sample_points,] %*% LFDAMetric
#' yhat2 <- knn(transformedX, transformednewX, subY, k = 5)
#' Accuracy2 <- length(which(Y[-sample_points] == yhat2))/length(Y[-sample_points])
#' Accuracy2
#'
#' @export
LFDA <- function(Y, X, metric = "plain", total_dims = NULL){

  # metric must be one of plain, weighted or orthogonalized!
  if(length(metric) != 1 | !(metric %in% c("plain", "orthogonalized", "weighted"))){
    stop("metric must be one of plain, weighted or orthogonalized!")
  }

  # Check the inputs X and Y
  labels <- sort(unique(Y))
  check_YAndX(Y, X, labels, check_pairs = FALSE)

  # total_dims must be a positive integer if default is not used
  if(length(total_dims) == 0 & is.null(total_dims)){
    total_dims <- ncol(X)
  } else if(length(total_dims) != 1 | floor(total_dims) != total_dims | total_dims <= 0 | total_dims > ncol(X)){
    stop("total_dims must be a positive integer less than or equal to ncol(X) and greater or equal than 1!")
  }

  results <- lfda(Y, X, labels, total_dims, metric)
  return(results)

}
