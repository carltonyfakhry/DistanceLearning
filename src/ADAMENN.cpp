#include "Utils.h"



/**
 *
 * References:
 *
 * [1] C. Domeniconi, D. Gunopulos, Locally adaptive metric nearest-neighbor
 * classification, IEEE Transactions on Pattern Analysis and Machine Intelligence
 * 24 (2002) 1281–1285.
 *
 */



/**
 *
 * This function computes the neighborhood of a point using the distance D(x,y)
 * as defined in [1].
 *
 */

arma::uvec ADAMENNdist_neighborhood(const arma::mat &data, const arma::rowvec &point, const arma::rowvec &w, int k){

  // Compute the distances from the focal point
  arma::vec distances = arma::zeros(data.n_rows);

  for (unsigned int i = 0; i < data.n_rows; i++){
    arma::rowvec temp = arma::pow((data.row(i) - point),2);
    temp = w % temp;
    double dist = sqrt(arma::sum(temp));
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
 * This function computes the relevance of a point r(z) as defined
 * in [1].
 *
 */

arma::rowvec getr(const arma::uvec &Y, const arma::mat &X, const arma::rowvec &x_0,
                    const arma::rowvec w, const arma::uvec &labels, int K_1, int K_2, int L){

  arma::rowvec r = arma::zeros<arma::rowvec>(X.n_cols);

  arma::uvec N_1 = ADAMENNdist_neighborhood(X, x_0, w, K_1);
  arma::uvec N_2 = ADAMENNdist_neighborhood(X, x_0, w, K_2);

  arma::uvec Y_1 = Y.rows(N_1);
  arma::uvec Y_2 = Y.rows(N_2);
  arma::mat subX = X.rows(Y_2);

  for(unsigned int i = 0; i < X.n_cols; i++){

    arma::uvec feature(1);
    arma::rowvec x_0i(1);
    arma::vec w_i(1);
    feature[0] = i;
    x_0i[0] = x_0[i];
    w_i[0] = w[i];

    arma::mat subX_i = subX.cols(feature);
    arma::uvec N_L = ADAMENNdist_neighborhood(subX_i, x_0i, w_i, L);
    arma::uvec Y_L = Y_2.rows(N_L);

    for(unsigned int j = 0; j < labels.n_elem; j++){

      double P_hat1 = 0;
      double P_hat2 = 0;

      arma::uvec N_1_j = arma::find(Y_1 == j);
      if(Y_1.n_elem > 0){
        P_hat1 = N_1_j.n_elem / (Y_1.n_elem + 0.0);
      }

      arma::uvec N_L_j = arma::find(Y_L == j);
      if(Y_L.n_elem > 0){
        P_hat2 = N_L_j.n_elem / (Y_L.n_elem + 0.0);
      }

      if(P_hat2 > 0){
        r[i] += pow((P_hat1 - P_hat2),2) / P_hat2;
      }

    }

  }

  return r;

}



/**
 *
 * This function computes r^{hat}(x_0) as defined in [1].
 *
 */

arma::rowvec getrhat(const arma::uvec &Y, const arma::mat &X, const arma::rowvec &x_0,
                      const arma::rowvec w, const arma::uvec &labels, int K_0, int K_1, int K_2, int L){

  arma::uvec N_K = ADAMENNdist_neighborhood(X, x_0, w, K_0);
  arma::mat X_K = X.rows(N_K);
  arma::mat r_K = arma::mat(X_K.n_rows, X_K.n_cols);

  for(unsigned int i = 0; i < X_K.n_rows; i++){

    arma::rowvec z = X_K.row(i);
    r_K.row(i) = getr(Y, X, z, w, labels, K_1, K_2, L);

  }

  arma::rowvec rhat = arma::sum(r_K, 0);
  if(N_K.n_elem > 0){
    rhat = rhat / (N_K.n_elem + 0.0);
  }
  return rhat;

}



/**
 *
 * This function computes a vector R whose components are
 * R_i(x_0) = max(r^{hat}) - r^{hat}_i as defined in [1].
 *
 */

arma::rowvec getR(const arma::rowvec &rhat){

  arma::rowvec R(rhat.n_elem);
  double max_rhat = rhat.max();

  for(unsigned int i = 0; i < rhat.n_elem; i++){

    R[i] = max_rhat - rhat[i];

  }

  return R;

}



/**
 *
 * This function computes the weights w of the metric as
 * defined in equation (5) in [1].
 *
 */

arma::rowvec getw(arma::rowvec R, double c, double total){

  arma::rowvec w(R.n_elem);

  for(unsigned int i = 0; i < R.n_elem; i++){

    w[i] = exp(c * R[i]) / total;

  }

  return w;

}



/**
 *
 * This function computes ADAMENN as defined in Fig. 1 in [1].
 *
 */
// [[Rcpp::export]]
arma::ivec adamenn(const arma::uvec &Y, const arma::mat &X, const arma::mat &newX,
                    const arma::uvec &labels, int K, int K_0, int K_1, int K_2,
                    int L, double c, int n_iterations){

  arma::ivec predictions = -1*arma::ones<arma::ivec>(newX.n_rows);
  bool can_be_classified = true;

  for(unsigned int i = 0; i < newX.n_rows; i++){

    Rcpp::checkUserInterrupt();
    int iter = 1;
    arma::rowvec w = arma::ones<arma::rowvec>(newX.n_cols);
    arma::rowvec x_0 = newX.row(i);

    do{

      arma::rowvec rhat = getrhat(Y, X, x_0, w, labels, K_0, K_1, K_2, L);
      arma::rowvec R = getR(rhat);
      double total = arma::sum(arma::exp(c * R));
      if(std::isinf(total)){
        can_be_classified = false;
        break;
      }
      w = getw(R, c, total);
      iter++;

    } while (iter <= n_iterations);

    if(can_be_classified){

      // Predict using K-NN
      arma::uvec indices = ADAMENNdist_neighborhood(X, x_0, w, K);
      arma::uvec subY = Y.rows(indices);
      predictions[i] = KNN(subY, labels);

    }

  }

  return predictions;

}
