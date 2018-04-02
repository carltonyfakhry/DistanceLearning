#include "Utils.h"


/**
 *
 * [1] Dimensionality reduction of multimodal labeled data by
 * local Fisher discriminant analysis. Journal of Machine Learning
 * Research, vol.8, 1027--1061.
 *
 * [2] Local Fisher discriminant analysis for supervised dimensionality reduction.
 * In W. W. Cohen and A. Moore (Eds.), Proceedings of 23rd International
 * Conference on Machine Learning (ICML2006), 905--912.
 *
 */



/**
 *
 * Computing the lfda metric as presented in Figure 2 in [1].
 *
 */

arma::mat computeLFDAMetric(const arma::uvec &Y, const arma::mat &X,
                            const arma::uvec &labels,
                            int total_dims, std::string method){

  arma::mat lfda_metric;

  double n = X.n_rows;

  arma::mat S_b = arma::zeros(X.n_cols, X.n_cols);
  arma::mat S_w = arma::zeros(X.n_cols, X.n_cols);

  // Compute the scatter matrices
  for(unsigned int l = 0; l < labels.n_elem; l++){

    arma::uvec indices = arma::find(Y == labels[l]);
    arma::mat subX = X.rows(indices);
    arma::vec sigma = arma::vec(subX.n_rows);
    double n_l = subX.n_rows;

    // Compute sigma for each point in the class
    for(unsigned int i = 0; i < indices.n_elem; i++){

      Rcpp::List lst = get_neighborhood(subX, subX.row(i), 8); // Get the 8 closest data points including the data point itself
      arma::vec distances = lst["distances"];
      sigma[i] = distances.max();

    }

    // Compute the affinity matrix
    arma::mat A = arma::zeros(subX.n_rows, subX.n_rows);
    double dist = 0;

    for(unsigned int i = 0; i < n_l; i++){

      arma::rowvec point = subX.row(i);
      double sigma_i = sigma[i];

      for(unsigned int j = 0; j < n_l; j++){

        dist = arma::sum(arma::pow(point - subX.row(j),2));
        A(i,j) = exp(-dist/(sigma_i * sigma[j]));

      }

    }

    subX = subX.t();

    // Compute G
    arma::mat G = subX * arma::diagmat(A * arma::ones(subX.n_cols)) * subX.t() - subX * A * subX.t();

    arma::vec X1_nl = subX * arma::ones(subX.n_cols);

    S_b = S_b + G/n + (1 - n_l/n) * subX * subX.t() + X1_nl * X1_nl.t() / n;
    S_w = S_w + G/n_l;


  }

  arma::mat X1_n = X.t() * arma::ones(X.n_rows);
  S_b = S_b - X1_n * X1_n.t() / n - S_w;

  // Make sure S_b and S_w are symmetric
  S_b = (S_b + S_b.t())/2;
  S_w = (S_w + S_w.t())/2;

  // Compute the metric
  try{

    arma::cx_vec eigvals;
    arma::cx_mat eigvecs;

    bool success = arma::eig_pair(eigvals, eigvecs, S_b, S_w);

    if(!success){
      Rcpp::stop("The given generalized eivenvalue problem is not solvable. Metric cannot be computed!");
    }

    lfda_metric = arma::conv_to<arma::mat>::from(eigvecs);

    // Convert metric according to the method requested
    if(method == "weighted"){

      arma::vec weights = arma::conv_to<arma::vec>::from(eigvals);

      for(unsigned int i = 0; i < weights.n_elem; i++){

        if(weights[i] >= 0){

          lfda_metric.col(i) = lfda_metric.col(i) * sqrt(weights[i]);

        }else{

          // lfda_metric.col(i) = lfda_metric.col(i) * 0.0;
          Rcpp::stop("Cannot compute the weighted metric since some eigenvalues are negative!");

        }

      }


    }else if(method == "orthogonalized"){

      arma::mat Q, R;
      bool success = arma::qr(Q, R, lfda_metric);

      if(!success){

        Rcpp::stop("Orthogonalization failed, try to use a different type of metric (i.e either plain or weighted)!");

      }

      lfda_metric = Q;

    }

  }catch(std::runtime_error &e){

    Rcpp::Rcout << e.what() << "\n";
    Rcpp::stop("The given generalized eivenvalue problem is not solvable. Metric cannot be computed!");

  }

  return lfda_metric;

}



/**
 *
 * This function computes the lfda metric.
 *
 */
// [[Rcpp::export]]
Rcpp::List lfda(const arma::uvec &Y, const arma::mat &X,
                const arma::uvec &labels, int total_dims, std::string method){

  arma::mat lfda_metric = computeLFDAMetric(Y, X, labels, total_dims, method);
  lfda_metric = lfda_metric.head_cols(total_dims);

  arma::mat TransformedX = X * lfda_metric;

  return Rcpp::List::create(Rcpp::Named("LFDATransform") = lfda_metric,
                            Rcpp::Named("transformedX") = TransformedX);

}
