#include "Utils.h"



/**
 *
 * [1] J. Goldberger, S. Roweis, G. Hinton, and R. Salakhutdinov. Neighbourhood components analysis.
 * In L. K. Saul, Y. Weiss, and L. Bottou, editors, Advances in Neural Information Processing
 * Systems 17, pages 513–520, Cambridge, MA, 2005. MIT Press.
 *
 */



/**
 *
 * This function computes the neighborhood component analysis (NCA) algorithm as described
 * in [1]. The algorithm was adapted from https://github.com/jhseu/nca.
 *
 */
// [[Rcpp::export]]
Rcpp::List nca(const arma::uvec &Y, const arma::mat &X,
               const arma::uvec &labels, unsigned int iterations, double learning_rate){

  // Initialize A
  arma::mat A = arma::diagmat(1.0/(arma::max(X, 0) - arma::min(X, 0)));
  for(unsigned int i = 0; i < A.n_rows; i++){
    for(unsigned int j = 0; j < A.n_cols; j++){
      if(std::isnan(A(i,j)) || std::isinf(A(i,j))){
        A(i,j) = 0;
      }
    }
  }


  for(unsigned int iter = 0; iter < iterations; iter++){

    unsigned int i = iter % Y.n_elem;
    arma::vec point = X.row(i).t();

    // For a given point, compute the softmax as defined in [1]
    arma::vec softmax = arma::zeros(X.n_rows);

    for(unsigned int k = 0; k < X.n_rows; k++) {

      if(k == i) continue;

      softmax[k] = std::exp(-std::pow(arma::norm(A*point - A*X.row(k).t()), 2));

    }

    double softmax_normalization = arma::sum(softmax);
    if(softmax_normalization > 0){

        softmax = softmax / arma::sum(softmax);

    }

    // Compute the probability that the point will be correctly
    // classified
    arma::vec softmax_simlabs = softmax.rows(arma::find(Y == Y[i]));
    double probability = arma::sum(softmax_simlabs);

    // Compute the gradient
    arma::mat first_term = arma::zeros(X.n_cols, X.n_cols);
    arma::mat second_term = arma::zeros(X.n_cols, X.n_cols);

    for(unsigned int k = 0; k < X.n_rows; ++k) {

      if(k == i) continue;

      arma::vec x_ik = point - X.row(k).t();
      arma::mat term = softmax[k] * (x_ik * x_ik.t());
      first_term += term;

      if(Y[k] == Y[i]){

        second_term += term;

      }

    }

    first_term *= probability;
    A += learning_rate*A*(first_term - second_term);

  }

  // Transform the data
  arma::mat transformedX = X * A;

  return Rcpp::List::create(Rcpp::Named("NCATransform") = A,
                            Rcpp::Named("transformedX") = transformedX);

}
