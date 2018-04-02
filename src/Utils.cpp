#include "Utils.h"



/**
 *
 *  Count the number of distinct pairs of points with different
 *  class labels, without taking order into account.
 *
 */

unsigned int getPairsCount(const arma::uvec &testY, const arma::uvec &labels){

  unsigned int total = 0;

  for(unsigned int i = 0; i < labels.size()-1; i++){

    arma::uvec labs1 = arma::find(testY == labels[i]);
    arma::uvec labs2 = arma::find(testY > labels[i]);
    total += labs1.size()*labs2.size();

  }

  if(std::isinf(total))
    Rcpp::stop("The number of pairs of points with different class labels is infinite. Please specify a smaller dissimilarity matrix!");


  return total;

}



/**
 *
 * This function returns indices for points in the local neighborhood
 * (euclidean distance) of a given focal point and their respective
 * distances.
 *
 */

Rcpp::List get_neighborhood(arma::mat data, arma::rowvec point, unsigned int k){

  // Compute the distances from the focal point
  arma::vec distances = arma::zeros(data.n_rows);

  for (unsigned int i = 0; i < data.n_rows; i++){
    double dist = sqrt(arma::sum(arma::pow((data.row(i) - point),2)));
    distances[i] = dist;
  }

  // Get the ranks from smallest to largest distance
  arma::uvec ranks_for_sorted = arma::sort_index(arma::sort_index(distances, "ascend"));

  // Get the indices for the closest k points to a focal point
  arma::uvec indices = arma::find(ranks_for_sorted < k);


  // Get the distances of the rows with selected indices
  distances = distances.rows(indices);

  return Rcpp::List::create(Rcpp::Named("indices") = indices,
                            Rcpp::Named("distances") = distances);

}



/**
 *
 * This function returns indices for points in the local neighborhood
 * (euclidean distance) of a given focal point.
 *
 */

arma::uvec get_neighborhood2(const arma::mat &data, const arma::rowvec &point, unsigned int k){

  // Compute the distances from the focal point
  arma::vec distances = arma::zeros(data.n_rows);

  for (unsigned int i = 0; i < data.n_rows; i++){
    double dist = sqrt(arma::sum(arma::pow((data.row(i) - point),2)));
    distances[i] = dist;
  }

  // Get the ranks from smallest to largest distance
  arma::uvec ranks_for_sorted = arma::sort_index(arma::sort_index(distances, "ascend"));

  // Get the indices for the closest k points to a focal point
  arma::uvec indices = arma::find(ranks_for_sorted < k);

  return indices;

}




/**
 *
 * This function computes the pseudo inverse of a psd square matrix.
 *
 */

arma::mat getPseudoInverse(const arma::mat &X, double tolerance){

  arma::mat U;
  arma::vec s;
  arma::mat V;
  bool success = arma::svd(U, s, V, X);

  if(!success){

    success = arma::svd(U, s, V, X, "std");

    if(!success){

      throw std::runtime_error("The matrix has no inverse and no pseudo-inverse!");

    }

  }

  // arma::uvec indices = arma::find(s > 0);
  arma::uvec indices = arma::find(s > std::max(tolerance*s[0],0.0));
  if(indices.n_elem == 0){

    throw std::runtime_error("The matrix has no inverse and no pseudo-inverse!");

  }

  s = s.rows(indices);
  U = U.cols(indices);
  V = V.cols(indices);

  int cutoff = 0;
  for(unsigned int i = 0; i < s.n_elem; i++){

    double val = 1.0/sqrt(s[i]);

    if(std::isinf(val)){

      break;

    }else{

      cutoff++;
      s[i] = val;

    }

  }

  if(cutoff == 0){

    throw std::runtime_error("The matrix has no inverse and no pseudo-inverse!");

  }

  U = U.head_cols(cutoff);
  s = s.head_rows(cutoff);
  V = U.head_cols(cutoff);
  arma::mat s_neg_1_2 = arma::diagmat(s);
  arma::mat X2 = U * s_neg_1_2 * V.t();

  return(X2);

}



/**
 *
 * This function computes the negative square root of a psd symmetric square
 * matrix.
 *
 */

arma::mat NegSqrtMat(const arma::mat &X, double tolerance){

  arma::mat U;
  arma::vec s;
  arma::mat V;
  bool success = arma::svd(U, s, V, X);

  if(!success){

    success = arma::svd(U, s, V, X, "std");

    if(!success){

      throw std::runtime_error("The negative square root of a positive semidefinite matrix cannot be computed!");

    }

  }

  // arma::uvec indices = arma::find(s > 0);
  arma::uvec indices = arma::find(s > std::max(tolerance*s[0],0.0));
  if(indices.n_elem == 0){

    throw std::runtime_error("The negative square root of a positive semidefinite matrix cannot be computed!");

  }

  s = s.rows(indices);
  U = U.cols(indices);
  V = V.cols(indices);

  int cutoff = 0;
  for(unsigned int i = 0; i < s.n_elem; i++){

    double val = 1.0/sqrt(s[i]);

    if(std::isinf(val)){

      break;

    }else{

      cutoff++;
      s[i] = val;

    }

  }

  if(cutoff == 0){

    throw std::runtime_error("The negative square root of a positive semidefinite matrix cannot be computed!");

  }

  U = U.head_cols(cutoff);
  s = s.head_rows(cutoff);
  V = U.head_cols(cutoff);
  arma::mat s_neg_1_2 = arma::diagmat(s);
  arma::mat X2 = U * s_neg_1_2 * V.t();

  return(X2);
}



/**
 *
 * This function computes the square root of a psd symmetric square
 * matrix.
 *
 */

arma::mat SqrtMat(const arma::mat &X, double tolerance){

  arma::mat U;
  arma::vec s;
  arma::mat V;
  bool success = arma::svd(U, s, V, X);

  if(!success){

    success = arma::svd(U, s, V, X, "std");

    if(!success){

      throw std::runtime_error("The square root of a positive semidefinite matrix cannot be computed!");

    }

  }

  // arma::uvec indices = arma::find(s > 0);
  arma::uvec indices = arma::find(s > std::max(tolerance*s[0],0.0));
  if(indices.n_elem == 0){

    throw std::runtime_error("The square root of a positive semidefinite matrix cannot be computed!");

  }

  s = s.rows(indices);
  U = U.cols(indices);
  V = V.cols(indices);
  s = arma::sqrt(s);
  arma::mat s_neg_1_2 = arma::diagmat(s);
  arma::mat X2 = U * s_neg_1_2 * V.t();

  return(X2);

}



/**
 *
 * This function computes the means of points in each class label.
 *
 */

arma::mat getMeans(const arma::uvec &Y, const arma::mat &X, const arma::uvec &labels){

  arma::mat means = arma::zeros(labels.n_elem, X.n_cols);
  arma::mat subX;
  arma::rowvec mean;
  arma::uvec indices;

  for(unsigned int i = 0; i < labels.n_elem; i++){
    indices = arma::find(Y == labels[i]);
    subX = X.rows(indices);
    mean = arma::conv_to<arma::rowvec>::from(arma::mean(subX));
    means.row(i) = mean;
  }

  return means;

}



/**
 *
 * Predict labels using K-NN.
 *
 */

int KNN(const arma::uvec &Y, const arma::uvec &labels){

  int best_label = -1;
  unsigned int best_count = 0;

  for(unsigned int i = 0; i < labels.n_elem; i++){

    arma::uvec temp = arma::find(Y == labels[i]);
    if(temp.n_elem > best_count){
      best_count = temp.n_elem;
      best_label = labels[i];
    }

  }

  return best_label;

}

