#include "Utils.h"


/**
 *
 * References:
 *
 * [1] Aharon Bar-Hillel, Tomer Hertz, Noam Shental, and Daphna Weinshall. Learning distance
 * functions using equivalence relations. In In Proceedings of the Twentieth International
 * Conference on Machine Learning, pages 11–18.
 *
 * [2] N. Shental, T. Hertz, D. Weinshall, and M. Pavel. Adjustment learning and relevant
 * component analysis. In Proceedings of the Seventh European Conference on Computer Vision,
 * volume 4, pages 776–792, Copenhagen, Denmark, 2002.
 *
 */



/**
 *
 * Computing Chat as indicated in equation (1) of reference [1].
 *
 */

arma::mat getChat(const arma::mat &X, const arma::uvec &chunklets,
                    const arma::uvec &unique_chunklets, const arma::mat &means){

  arma::mat Chat = arma::zeros(X.n_cols, X.n_cols);

  for(unsigned int i = 0; i < unique_chunklets.n_elem; i++){

    arma::uvec indices = arma::find(chunklets == unique_chunklets[i]);
    arma::mat subX = X.rows(indices);
    arma::rowvec mean = means.row(i);

    for(unsigned int j = 0; j < subX.n_rows; j++){

      arma::rowvec diff = subX.row(j) - mean;
      Chat = Chat + diff.t() * diff;

    }

  }

  Chat = Chat / chunklets.n_elem;

  return Chat;

}



/**
 *
 * This function computes RCA metric and transforms the data according to it.
 * Predictions are made for the new data points according to KNN after the transformation of the data.
 *
 */
// [[Rcpp::export]]
Rcpp::List rca(const arma::mat &X,
                const arma::ivec &chunklets, const arma::uvec &unique_chunklets, int total_dims){

  arma::uvec indices = arma::find(chunklets != -1);
  arma::mat subX = X.rows(indices);
  arma::uvec subchunklets = arma::conv_to<arma::uvec>::from(chunklets.rows(indices));
  arma::mat means = getMeans(subchunklets, subX, unique_chunklets);
  arma::mat Chat = getChat(subX, subchunklets, unique_chunklets, means);
  arma::mat Chat_neg_1_2;

  // Make sure the negative square root of a metric can be obtained
  try {

    Chat_neg_1_2 = NegSqrtMat(Chat);

  }catch(std::runtime_error &e){

    Rcpp::stop("The metric could not be computed!");

  }

  Chat_neg_1_2 = Chat_neg_1_2.head_cols(total_dims);
  arma::mat transformedX = X * Chat_neg_1_2;

  return Rcpp::List::create(Rcpp::Named("RCATransform") = Chat_neg_1_2,
                            Rcpp::Named("transformedX") = transformedX);

}

