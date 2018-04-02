#' This function computes the Free Energy Metric Learning (FENN)
#' classification algorithm.
#'
#' @description  This function computes the Von-Neumann Metric Learning (FENN)
#'              classification algorithm as described in [1]. See the Vignette
#'              by using the command \strong{browseVignette("DistanceLearning")}
#'              for an introduction to using FENN.
#'
#' @usage FENN(Y, X, method = "CV", CV_window = NA, dimension_reduction = FALSE,
#'             threshold = 0.99, K = 5)
#'
#' @param Y vector of non-negative integer labels corresponding to each data point.
#' @param X Input numeric matrix where each row is a data point whose
#'          label is the corresponding entry in \code{Y} and each column is a
#'          variable.
#' @param method The method for estimating the smoothing parameter This can either be set
#'               to \code{"CV"} or \code{"fisher.information"}. The default value is
#'               \code{method = "CV"} in which case the smoothing parameter is determined
#'               using 10-fold cross-validation. The possible values for the smoothing
#'               parameter are generated in a window around the value that maximizes the
#'               fisher information.
#'               If \code{method = "fisher.information"} then the smoothing parameter is
#'               selected according to the value that maximizes the fisher information.
#' @param CV_window The window around the value of the smoothing parameter that maximizes the fisher
#'                  information. The default value is set to \code{CV_window = NA} in
#'                  which case the value \code{CV_window = ncol(X)} is set internally.
#'                  If \code{method = "fisher.information"} then the value of this parameter
#'                  is not considered when computing the FENN metric.
#' @param dimension_reduction Learn the transformation matrix for the optimal dimension
#'                            of the reduced space. The default value is
#'                            \code{dimension_reduction = FALSE}.
#' @param threshold The threshold cutoff on the sum of free energies when performing
#'                  dimension reduction. The default value is \code{threshold = 0.99}.
#' @param K The number of neighbors to be used for K-NN classification during the CV
#'          procedure when computing the optimal tuning parameter. The user is advised
#'          to set this value to the number of neighbors that will be used in classification
#'          in the transformed space after the FENN metric is applied to the original data.
#'          Default value is set to \code{K = 5}.
#'
#' @return This function returns a list with the following items:
#' \item{FENNTransform}{The matrix under which the data was transformed. The
#'               multiplication of this matrix with its transpose gives
#'               the matrix used in the Mahalanobis
#'               metric.}
#' \item{TransformedX}{The transformed original data \strong{X} which was transformed
#'                     using the FENN trasformation matrix i.e \eqn{TransformedX = X *
#'                     FENNTransform}}.
#'
#' @details See the Vignette by using the command
#'          \strong{browseVignette("DistanceLearning")}
#'          for an introduction to using FENN.
#'
#' @author Carl Tony Fakhry, Ping Chen, Rahul Kulkarni and Kourosh Zarringhalam
#'
#' @references [1] C. T. Fakhry, P. Chen, R. Kularni and K. Zarringhalam.
#'             A Free Energy Based Approach for Distance Metric Learning, Submitted.
#'
#' @examples # Load data from package DistanceLearning
#' library(DistanceLearning)
#' fname <- system.file("extdata", "example_data.csv", package="DistanceLearning")
#' df <- read.csv(fname)
#' Y <- as.integer(df$y)
#' X <- as.matrix(df[,c(2,3)])
#' sample_points <- sample(1:nrow(X), 40, replace = FALSE)
#' trainX <- X[sample_points,]
#' trainY <- Y[sample_points]
#' testX <- X[-sample_points,]
#' testY <- Y[-sample_points]
#'
#' # Learn the metric, and get the transformed data
#' result <- FENN(trainY, trainX, method = "CV")
#' FENNTransform <- result$FENNTransform
#'
#' # Get the accuracy of KNN classification without applying the new metric
#' yhat <- knn(trainX, testX, trainY, k = 10, prob = FALSE)
#' Accuracy <- length(which(testY == yhat))/length(testY)
#' Accuracy
#'
#' # Get the accuracy of KNN classification after applying the new metric
#' transformedtestX <- testX %*% FENNTransform
#' transformedtrainX <- trainX %*% FENNTransform
#' yhat2 <- knn(transformedtrainX, transformedtestX, trainY, k = 5, prob = FALSE)
#' Accuracy2 <- length(which(testY == yhat2))/length(testY)
#' Accuracy2
#'
#' @export
FENN <- function(Y, X, method = "CV",
                 CV_window = NA, dimension_reduction = FALSE, threshold = 0.99, K = 5){

  # Check the inputs X, Y
  labels <- sort(unique(Y))
  check_YAndX(Y, X, labels, check_pairs = TRUE)

  # Check method
  if(!is.character(method) | length(method) != 1 | !any(method %in% c("CV", "fisher.information"))){
    stop("Method must be either CV or fisher.information!")
  }

  # Check threshold
  if(!is.numeric(threshold) | length(threshold) != 1 | threshold > 1 | threshold <= 0){
    stop("Threshold parameter must be <= 1 and > 0!")
  }

  # dimension_reduction must be a boolean
  if(length(dimension_reduction) != 1 | !is.logical(dimension_reduction)){
    stop("dimension_reduction must be a boolean value!")
  }

  # Check CV_window
  if(method == "CV"){
    if(length(CV_window) == 1 & is.na(CV_window)){
      CV_window <- 4 * nrow(X)
    }else if(!is.numeric(CV_window) | length(CV_window) != 1 | CV_window != floor(CV_window) | CV_window <= 0){
      stop("CV_window must be an integer > 0!")
    }
  }

  # K must be a positive integer
  if(length(K) != 1 | floor(K) != K | K <= 0){
    stop("K must be a positive integer!")
  }

  results <- fenn(Y, X, labels)
  flds <- createFolds(Y, k = 10, list = TRUE, returnTrain = FALSE)

  X_D_neg_1_2 = results$X_D_neg_1_2
  eigvecs = results$SimilarityDirections
  t_eigvecs = t(eigvecs)
  fisher_information = as.vector(results$FisherInformation)
  Mus = as.vector(results$MuVec)
  FreeEnergies = results$FreeEnergies
  mu_lambdas = results$MuLambdas
  mu_lambdas[mu_lambdas < 0] = 0
  mu_lambdas = sqrt(mu_lambdas)
  mu_index = NULL
  mu = NULL
  IndexBestDimensions = rep(0, length(Mus))

  accuracies <- c()
  if(method == "CV"){

    total_sum_lambdas = apply(mu_lambdas, 1, sum)
    inds_notzero = which(total_sum_lambdas > 0 & !is.infinite(total_sum_lambdas) & !is.na(total_sum_lambdas))
    mu_lambdas = mu_lambdas[inds_notzero,]
    Mus = Mus[inds_notzero]
    fisher_information = fisher_information[inds_notzero]
    FreeEnergies = FreeEnergies[inds_notzero,]
    max_fisher_ind = which.max(fisher_information)
    MuSearchGrid = (max(1,max_fisher_ind-CV_window)):(min(nrow(mu_lambdas),max_fisher_ind + CV_window))
    mu_lambdas = mu_lambdas[MuSearchGrid,]
    Mus = Mus[MuSearchGrid]
    fisher_information = fisher_information[MuSearchGrid]
    FreeEnergies = FreeEnergies[MuSearchGrid,]

    if(dimension_reduction){
      for(i in 1:nrow(FreeEnergies)){
        mu = Mus[i]
        free_energies = FreeEnergies[i,]
        thresh2 = mu * log(sum(free_energies))
        vals = rev(mu*log(cumsum(rev(free_energies))))
        inds = which(!is.na(vals) & !is.infinite(vals) & vals/thresh2 >= threshold)
        IndexBestDimensions[i] = max(inds)
      }
    }

    # 10 fold CV
    outputhat = c()
    output = matrix(, ncol = nrow(mu_lambdas), nrow = 0)
    for(j in 1:10){
      outputhat = c(outputhat, Y[flds[[j]]])
      output = rbind(output, matrix(-1, nrow = length(flds[[j]]), ncol = nrow(mu_lambdas)))
    }
    metric0 = X_D_neg_1_2 %*% eigvecs
    total = 0
    inds = 1:ncol(X)
    metric = NULL
    for(i in 1:10){
      testindices = flds[[i]]
      trainY = Y[-testindices]
      trainX = X[-testindices,]
      testY = Y[testindices]
      testX = X[testindices,]
      total = total + length(testindices)
      for(j in 1:nrow(mu_lambdas)){
        lambdas = mu_lambdas[j,]
        metric = metric0 %*% diag(lambdas, ncol = length(lambdas), nrow = length(lambdas)) %*% t_eigvecs
        if(dimension_reduction){
          index = IndexBestDimensions[j]
          metric = metric %*% eigvecs[,index:ncol(eigvecs),drop=F]
        }
        transformedtestX = testX %*% metric
        transformedtrainX = trainX %*% metric
        yhat = try(knn(transformedtrainX, transformedtestX, trainY, k = K))
        if(inherits(yhat, "try-error")) next
        yhat = as.integer(as.character(yhat))
        output[(total-length(testindices) + 1):total,j] = yhat
      }
    }
    accuracies = apply(output, 2, function(x,outputhat){length(which(outputhat == x))/length(x)}, outputhat = outputhat)
    max_indices = which(accuracies == max(accuracies))
    max_fisher_ind = which.max(fisher_information)
    min_dist_index = which.min(abs(max_fisher_ind - max_indices))
    mu_index = max_indices[min_dist_index[1]]
    mu = Mus[mu_index]

  }else if(method == "fisher.information"){

    if(dimension_reduction){
      for(i in 1:nrow(FreeEnergies)){
        mu = Mus[i]
        free_energies = FreeEnergies[i,]
        thresh2 = mu * log(sum(free_energies))
        vals = rev(mu*log(cumsum(rev(free_energies))))
        inds = which(!is.na(vals) & !is.infinite(vals) & vals/thresh2 >= threshold)
        IndexBestDimensions[i] = max(inds)
      }
    }

    mu_index <- which.max(fisher_information)
    mu <- Mus[mu_index]

  }

  lambdas = mu_lambdas[mu_index,]
  best_dim_index = IndexBestDimensions[mu_index]
  results[["FENNTransform"]] <- X_D_neg_1_2 %*% (eigvecs %*% diag(lambdas) %*% t_eigvecs)

  if(dimension_reduction && ncol(X) > 1){
    results[["FENNTransform"]] <- results[["FENNTransform"]] %*% eigvecs[,best_dim_index:ncol(eigvecs),drop = F]
    results[["TransformedX"]] <- X %*% results[["FENNTransform"]]
  }

  results[["Lambdas"]] <- lambdas^2
  results[["Energies"]] <- results$Energies
  results[["BestMuIndex"]] <- mu_index
  results[["BestMu"]] <- mu
  results[["MuLambdas"]] <- mu_lambdas^2
  results[["FreeEnergies"]] <- NULL
  results[["FisherInformation"]] <- fisher_information
  results[["MuVec"]] <- Mus
  results[["Accuracies"]] <- accuracies
  return(results)

}

