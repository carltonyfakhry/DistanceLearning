# Check the data matrix X and its class labels Y
check_YAndX <- function(Y, X, labels, check_pairs = TRUE){

  # X must be a numeric matrix with at least one column
  if(!is.numeric(X) & !is.double(X) & !is.integer(X) & !is.matrix(X)){
    stop("X must be a integer, numeric or double matrix!")
  }

  if(ncol(X) < 1){
    stop("X must have at least one column!")
  }

  # Y must all be integers greater than 1
  if(!is.vector(Y)){
    stop("Y must be a vector!")
  }

  if(any(Y < 0)){
    stop("All class labels in Y must be non-negative integers!")
  }

  if(any(Y != floor(Y))){
    stop("All class labels in Y must be non-negative integers!")
  }

  # Y and X must have the same length
  if(nrow(X) != length(Y)){
    stop("The number of labels in Y must match the number of data points in X!")
  }

  # There must be at least two classes represented
  if(length(labels) < 2){
    stop(paste("There must be at least 2 class labels present in Y."))
  }

  # X must have at least 2 points with different class labels
  if(nrow(X) < 2){
    stop("X must have at least 2 data points with different class labels!")
  }

  # No data should be nan or inf
  if(any(is.na(X)) | any(is.na(Y))){
    stop("X and Y should not contain NA values!")
  }

  if(any(is.nan(X)) | any(is.nan(Y))){
    stop("X and Y should not contain nan values!")
  }

  if(any(is.infinite(X)) | any(is.infinite(Y))){
    stop("X, and Y should not contain infinite values!")
  }

  if(check_pairs){
    tabulations = as.vector(table(Y))

    if(length(which(tabulations > 0)) < 2){
      mess1 = "There must be at least one pair of points with"
      mess2 = "different class labels so that the method can be called properly!"
      stop(paste(mess1, mess2))
    }

    if(!any(tabulations > 1)){
      mess1 = "There must be at least one pair of points with"
      mess2 = "similar class labels so that the method can be called properly!"
      stop(paste(mess1, mess2))
    }
  }

}



# Check newX input
check_newX <- function(X, newX){

  # X must be a numeric matrix with at least one column
  if(!is.numeric(newX) & !is.double(newX) & !is.integer(newX) & !is.matrix(newX)){
    stop("newX must be a integer, numeric or double matrix!")
  }

  # newX must be a numeric matrix
  if(!is.matrix(newX)){
    stop("newX must be a numeric matrix")
  }

  # X and newX must have the same number of columns
  if(ncol(X) != ncol(newX)){
    stop("X and newX must have the same number of columns!")
  }

  # No data should be na nan or inf
  if(any(is.na(newX))){
    stop("newX should not contain NA values!")
  }

  if(any(is.nan(newX))){
    stop("newX should not contain nan values!")
  }

  if(any(is.infinite(newX))){
    stop("newX should not contain infinite values!")
  }

  if(nrow(newX) < 1){
    stop("newX must contain at least 1 data point!")
  }

}


# Check X and chunklet inputs
check_X_Chunklets <- function(X, chunklets){

  # X must be a numeric matrix with at least one column
  if(!is.numeric(X) & !is.double(X) & !is.integer(X) & !is.matrix(X)){
    stop("X must be a integer, numeric or double matrix!")
  }

  # X must be a numeric matrix
  if(!is.matrix(X)){
    stop("X must be a numeric matrix")
  }


  # No data should be na, nan or inf
  if(any(is.na(X))){
    stop("X should not contain NA values!")
  }

  if(any(is.nan(X))){
    stop("X should not contain nan values!")
  }

  if(any(is.infinite(X))){
    stop("X should not contain infinite values!")
  }

  if(nrow(X) < 1){
    stop("X must contain at least 1 data point!")
  }

  # Chunklets are marked by non-negative integers
  if(any(chunklets != floor(chunklets))){
    stop("All chunklet numbers must be integers greater or equal to -1. Data points that do not belong to a chunklet are assigned a value of -1!")
  }

  if(any(chunklets < -1)){
    stop("All chunklet numbers must be integers greater or equal to -1. Data points that do not belong to a chunklet are assigned a value of -1!")
  }

  if(all(chunklets == -1)){
    stop("No data points are assigned to any chunklets!")
  }

  if(length(chunklets) != nrow(X)){
    stop("length(chunklets) should be equal to nrow(X)!")
  }

}

