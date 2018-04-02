#' This function computes the Discriminant Component Analysis (RCA)
#' algorithm.
#'
#' @description This function computes the Discriminant Component Analaysis
#'              (DCA) algorithm as described in [1]. See the Vignette
#'              by using the command \code{browseVignette("DistanceLearning")}
#'              for an introduction to using DCA.
#'
#' @usage DCA(X, chunklets, D, total_dims = NULL)
#'
#' @param X Input numeric matrix where each row is a data point and each
#'          column is a variable.
#' @param chunklets An integer vector indicating to which chunklet each data
#'                  point belongs to. If a data point does not belong to
#'                  a chunklet, then its chunklet value should be set to -1
#'                  otherwise all chunklets should be represented by
#'                  a non-negative integer. All data points in the same chunklet
#'                  must have the same class label.
#' @param D A 0-1 symmetric matrix where a 1 at \code{D[i,j]} indicates that
#'          chunklets \code{i} and \code{j} share a negative constraint i.e chunklets
#'          \code{i} and \code{j} do not belong to the same class. The position of the
#'          chunklets in the rows and columns of \code{D} is in
#'          increasing order of the integers representing the chunklets. For
#'          example, if we have chunklet 1 and chunklet 2, then the first row
#'          and first column in \code{D} will reference chunklet 1, and
#'          the second row and second column in \code{D} will reference
#'          chunklet 2.
#' @param total_dims The number of dimensions on which the data should be projected
#'                   on after the DCA metric has been learned. Default value is set
#'                   to \code{total_dims = NULL} in which case the value
#'                   \code{total_dims = ncol(X)} is used as default.
#'
#' @return This function returns a list with the following items:
#' \item{DCATransform}{The transformation matrix.}
#' \item{TransformedX}{The transformed original data \code{X} which was transformed
#'                     using the DCA trasformation matrix i.e \eqn{TransformedX = X *
#'                     DCATransform}}.
#'
#' @details See the Vignette by using the command
#'          \code{browseVignette("DistanceLearning")}
#'          for an introduction to using DCA.
#'
#' @author Carl Tony Fakhry, Ping Chen, Rahul Kulkarni and Kourosh Zarringhalam
#'
#' @references [1] Hoi, S.C.H., Liu, W., Lyu, M.R., Ma, W.Y.: Learning
#'                 distance metrics with contextual constraints for image
#'                 retrieval. In: IEEE Conference on Computer Vision and
#'                 Pattern Recognition.
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
#' @seealso \code{\link{RCA}}
#'
#' @export
DCA <- function(X, chunklets, D, total_dims = NULL){

  # Check the inputs X
  check_X_Chunklets(X, chunklets)

  # total_dims must be a positive integer if default is not used
  if(length(total_dims) == 0 & is.null(total_dims)){
    total_dims <- ncol(X)
  } else if(length(total_dims) != 1 | floor(total_dims) != total_dims | total_dims <= 0 | total_dims > ncol(X)){
    stop("total_dims must be a positive integer less than or equal to ncol(X) and greater or equal than 1!")
  }

  unique_chunklets <- unique(chunklets)

  # D must contain only 0 and 1 values
  if(!is.matrix(D)){
    stop("D must be a matrix!")
  }

  if(!isSymmetric(D)){
    stop("D must be symmetric!")
  }

  if(any(D != 0 & D != 1)){
    stop("D must contain only 0 and 1 values!")
  }

  # D must be a square matrix with dimension equal to the number of unique chunklets
  if(nrow(D) != length(unique_chunklets) | ncol(D) != length(unique_chunklets)){
    stop("D must be a square matrix with dimension equal to the number of unique chunklets!")
  }

  # If all D is 0 then the user should RCA instead
  if(all(D == 0)){
    stop("D has only 0 values. User is advised to use the RCA function instead!")
  }

  # Set the diagonal elements to 0
  for(i in 1:ncol(D)){
    D[i,i] <- 0
  }

  results <- dca(X, chunklets, unique_chunklets, D, total_dims)
  return(results)

}
