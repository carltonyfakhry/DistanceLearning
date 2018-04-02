#include "Utils.h"



/**
 *
 * References:
 *
 * [1] T. Hastie, R. Tibshirani, Discriminant adaptive nearest neighbor classification,
 * IEEE Transactions on Pattern Analysis and Machine Intelligence 18 (1996) 607–616.
 *
 */



/**
 *
 * The vector d is defined as in [1] i.e d_i = ||Sigma^(1/2)(x_i - x_0)||.
 *
 */

arma::vec getd(const arma::mat &X, const arma::mat &SigmaSqrt, const arma::vec &x_0){

  arma::vec d = arma::zeros(X.n_rows);

  for(unsigned int i = 0; i < X.n_rows; i++){
    arma::vec x = arma::conv_to<arma::vec>::from(X.row(i));
    d[i] = arma::norm(SigmaSqrt * (x - x_0), 2);
  }

  return d;

}



/**
 *
 * This function computes k(x_i, x_0; Sigma, h) as defined in [1].
 *
 */

arma::vec getk(const arma::vec &d, double h){

  arma::vec k = arma::zeros(d.n_elem);
  for(unsigned int i = 0; i < d.n_elem; i++){
    double I_i = (fabs(d[i]) < h) ? 1 : 0;
    k[i] = pow((1 - pow(d[i]/h, 3)),3) * I_i;
  }

  return k;

}



/**
 *
 * This function computes the weighted between sum of squares
 * matrix B(x_0; Sigma_0, h) as defined in [1].
 *
 */

arma::mat getB(const arma::uvec &Y, const arma::mat &X, const arma::vec &k, const arma::uvec &labels, double total){

  arma::mat B = arma::zeros(X.n_cols, X.n_cols);
  arma::mat X_bar = arma::conv_to<arma::vec>::from(arma::mean(X));

  for(unsigned int i = 0; i < labels.n_elem; i++){
    arma::uvec temp = arma::find(Y == labels[i]);
    if(temp.n_elem == 0){
      continue;
    }
    arma::vec temp2 = k.rows(temp);
    double num = arma::sum(temp2);
    double Pi_i = num/total;
    arma::mat X_j = X.rows(temp);
    arma::vec X_bar_j = arma::conv_to<arma::vec>::from(arma::mean(X_j));
    arma::mat diff = X_bar_j - X_bar;
    B = B + Pi_i * diff * diff.t();
  }

  return B;

}



/**
 *
 * This function computes the weighted within sum of squares
 * matrix W(x_0; Sigma_0, h) as defined in [1].
 *
 */

arma::mat getW(const arma::uvec &Y, const arma::mat &X, const arma::vec &k, const arma::uvec &labels, double total){

  arma::mat W = arma::zeros(X.n_cols, X.n_cols);

  for(unsigned int i = 0; i < labels.n_elem; i++){
    arma::uvec temp = arma::find(Y == labels[i]);
    if(temp.n_elem == 0){
      continue;
    }
    arma::vec subk = k.rows(temp);
    arma::mat X_i = X.rows(temp);
    arma::rowvec X_bar_i = arma::conv_to<arma::rowvec>::from(arma::mean(X_i));
    for(unsigned int j = 0; j < X_i.n_rows; j++){
      arma::rowvec x_j = X_i.row(j);
      arma::rowvec diff = x_j - X_bar_i;
      W = W + subk[j] * diff.t() * diff;
    }
  }

  W = W / total;
  return W;

}



/**
 *
 * This function combines getW and getB to overlap similar operations
 * and speed up the computation.
 *
 */

Rcpp::List getWandB(const arma::uvec &Y, const arma::mat &X, const arma::vec &k, const arma::uvec &labels, double total){

  arma::mat B = arma::zeros(X.n_cols, X.n_cols);
  arma::mat W = arma::zeros(X.n_cols, X.n_cols);
  arma::rowvec X_bar = arma::conv_to<arma::rowvec>::from(arma::mean(X));

  for(unsigned int i = 0; i < labels.n_elem; i++){

    arma::uvec temp = arma::find(Y == labels[i]);
    if(temp.n_elem == 0){
      continue;
    }
    arma::vec subk = k.rows(temp);
    double num = arma::sum(subk);
    double Pi_i = num/total;
    arma::mat X_i = X.rows(temp);
    arma::rowvec X_bar_i = arma::conv_to<arma::rowvec>::from(arma::mean(X_i));
    arma::rowvec diff = X_bar_i - X_bar;

    // Continue to compute B
    B = B + Pi_i * diff.t() * diff;

    // Continue to compute W
    for(unsigned int j = 0; j < X_i.n_rows; j++){
      arma::rowvec x_j = X_i.row(j);
      arma::rowvec diff2 = x_j - X_bar_i;
      W = W + subk[j] * diff2.t() * diff2;
    }

  }

  W = W / total;

  return Rcpp::List::create(Rcpp::Named("W") = W,
                            Rcpp::Named("B") = B);

}



/**
 *
 * This function computes the Discriminant Adaptive Nearest Neighbor (DANN) classifier.
 * K, K_M and epsilon are defined as in [1]. X and Y are the data points
 * and their respective labels. newX is the new data for which the labels must be
 * assigned. As described in [1], n_iterations is the number of times the metric
 * for a given neighborhood of a point will be computed. DANN is when n_iterations is
 * set to 1. When n_iterations is greater than 1, the term i-DANN is used.
 *
 */
// [[Rcpp::export]]
arma::ivec dann(arma::uvec &Y, arma::mat &X, arma::mat &newX, int K, int K_M, double epsilon, int n_iterations){

  arma::mat identity = arma::eye<arma::mat>(X.n_cols, X.n_cols);
  arma::ivec predictions = -1*arma::ones<arma::ivec>(newX.n_rows);
  arma::uvec labels = arma::unique(Y);
  arma::mat means = getMeans(Y, X, labels);

  for(unsigned int i = 0; i < newX.n_rows; i++){

    // Check for use interrupt
    Rcpp::checkUserInterrupt();

    bool can_be_classified = true;
    int iter = 1;
    arma::mat B;
    arma::mat W;
    arma::mat Sigma = arma::eye<arma::mat>(X.n_cols, X.n_cols);
    arma::mat SigmaSqrt = arma::eye<arma::mat>(X.n_cols, X.n_cols);

    do{

      arma::mat transX = X * SigmaSqrt.t();
      arma::rowvec row_x_0 = newX.row(i);
      row_x_0 = row_x_0 * SigmaSqrt.t();
      arma::vec x_0 = arma::conv_to<arma::vec>::from(row_x_0);
      arma::uvec indices = get_neighborhood2(transX, row_x_0, K_M);
      arma::mat subX = transX.rows(indices);
      arma::uvec subY = Y.rows(indices);

      arma::vec d = getd(subX, SigmaSqrt, x_0);

      double h = d.max();

      if(h <= 0){
        can_be_classified = false;
        break;
      }

      arma::vec k = getk(d, h);
      double total = arma::sum(k);

      if(total <= 0){
        can_be_classified = false;
        break;
      }

      Rcpp::List lst = getWandB(subY, subX, k, labels, total);

      arma::mat B = lst["B"];
      arma::mat W = lst["W"];
      arma::mat WNegSqrt;

      try {

        WNegSqrt = NegSqrtMat(W);

      }catch(std::runtime_error &e){

        can_be_classified = false;
        break;

      }

      Sigma = WNegSqrt * (WNegSqrt * B * WNegSqrt + epsilon * identity) * WNegSqrt;

      try {

        SigmaSqrt = SqrtMat(Sigma);

      }catch(std::runtime_error &e){

        can_be_classified = false;
        break;

      }

      iter++;

    } while(iter <= n_iterations);

    // Transform the data and predict using K-NN
    if(can_be_classified){

      arma::mat transX = X * SigmaSqrt.t();
      arma::rowvec row_x_0 = newX.row(i);
      row_x_0 = row_x_0 * SigmaSqrt.t();
      arma::uvec indices = get_neighborhood2(transX, row_x_0, K);
      arma::uvec subY = Y.rows(indices);

      // Predict using K-NN
      predictions[i] = KNN(subY, labels);

    }

  }

  return predictions;

}
