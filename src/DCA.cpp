#include "Utils.h"



/**
 *
 * References:
 *
 * [1] Hoi, S.C.H., Liu, W., Lyu, M.R., Ma, W.Y.: Learning distance
 * metrics with contextual constraints for image retrieval. In: IEEE
 * Conference on Computer Vision and Pattern Recognition.
 *
 */



/**
 *
 * Compute C_hat_b in equation (3) in [1].
 *
 */

arma::mat getC_hat_b(const arma::mat &means, const arma::umat &D){

  int n_b = 0;
  arma::mat C_hat_b = arma::zeros(means.n_cols, means.n_cols);

  for(unsigned int i = 0; i < means.n_rows; i++){

    arma::uvec D_i = arma::conv_to<arma::uvec>::from(D.row(i));
    arma::uvec indices = arma::find(D_i == 1);
    arma::rowvec mean = means.row(i);
    n_b += indices.n_elem;

    for(unsigned int j = 0; j < indices.n_elem; j++){

      arma::rowvec diff = mean - means.row(indices[j]);
      C_hat_b = C_hat_b + diff.t() * diff;

    }

  }

  if(n_b > 0){
    C_hat_b = C_hat_b / (n_b + 0.0);
  }

  return C_hat_b;

}



/**
 *
 * Compute C_hat_w in equation (3) in [1].
 *
 */

arma::mat getC_hat_w(const arma::mat &X, const arma::ivec &chunklets, const arma::uvec &unique_chunklets,
                      const arma::mat &means){

  arma::mat C_hat_w = arma::zeros(X.n_cols, X.n_cols);

  for(unsigned int i = 0; i < unique_chunklets.n_elem; i++){

    arma::uvec indices = arma::find(chunklets == unique_chunklets[i]);
    arma::mat subX = X.rows(indices);
    arma::rowvec mean = means.row(i);

    for(unsigned int j = 0; j < subX.n_rows; j++){

      arma::rowvec diff = subX.row(j) - mean;
      C_hat_w = C_hat_w + (1/(indices.n_elem + 0.0)) * (diff.t() * diff);

    }

  }

  C_hat_w = C_hat_w / (unique_chunklets.n_elem + 0.0);

  return C_hat_w;

}



/**
 *
 * This function computes the DCA algorithm as described in [1].
 *
 */
// [[Rcpp::export]]
Rcpp::List dca(const arma::mat &X,
                const arma::ivec &chunklets, const arma::uvec &unique_chunklets,
                const arma::umat &D, int total_dims){

  arma::uvec temp = arma::find(chunklets != -1);
  arma::mat subX = X.rows(temp);
  arma::uvec subchunklets = arma::conv_to<arma::uvec>::from(chunklets.rows(temp));
  arma::mat means = getMeans(subchunklets, subX, unique_chunklets);

  // Compute C_hat_w and C_hat_b
  arma::mat C_hat_w = getC_hat_w(X, chunklets, unique_chunklets, means);
  arma::mat C_hat_b = getC_hat_b(means, D);

  // Compute D_b
  arma::mat U;
  arma::vec s;
  arma::mat V;

  try{

    bool success = arma::svd(U, s, V, C_hat_b);

    if(!success){

      success = arma::svd(U, s, V, C_hat_b, "std");

      if(!success){

        Rcpp::stop("Metric cannot be computed!");

      }

    }

  } catch(std::runtime_error &e){

    Rcpp::Rcout << e.what() << "\n";
    Rcpp::stop("Metric cannot be computed!");

  }

  arma::uvec indices = arma::find(s > 0);

  // No suitable eigendecomposition found for C_hat_b
  if(indices.n_elem == 0){
    Rcpp::stop("Metric cannot be computed!");
  }

  // U = U.head_cols(indices.n_elem);

  // Compute D_b
  arma::mat D_b = U.t() * C_hat_b * U;

  // Compute D_b_neg_1_2
  arma::mat D_b_neg_1_2;

  try{

    D_b_neg_1_2 = NegSqrtMat(D_b);

  } catch(std::runtime_error &e){

    Rcpp::Rcout << e.what() << "\n";
    Rcpp::stop("Metric cannot be computed!");

  }

  // Compute Z
  arma::mat Z = U * D_b_neg_1_2;

  // Compute C_z
  arma::mat C_z = Z.t() * C_hat_w * Z;

  arma::mat U2;
  arma::vec s2;
  arma::mat V2;
  arma::svd(U2, s2, V2, C_z);

  // Do dimension reduction
  U2 = U2.head_cols(total_dims);

  arma::mat D_w = U2.t() * C_z * U2;

  // Compute D_w_neg_1_2
  arma::mat D_w_neg_1_2;

  try{

    D_w_neg_1_2 = NegSqrtMat(D_w);

  } catch(std::runtime_error &e){

    Rcpp::Rcout << e.what() << "\n";
    Rcpp::stop("Metric cannot be computed!");

  }

  // Compute A
  arma::mat A = Z * U2 * D_w_neg_1_2;


  arma::mat transformedX = X * A;

  return Rcpp::List::create(Rcpp::Named("DCATransform") = A,
                            Rcpp::Named("transformedX") = transformedX);

}
