#include "Utils.h"



/**
 *
 * This function computes the negative square root of the dissimilarity matrix i.e X_D_neg_1_2.
 *
 */
// [[Rcpp::export]]
arma::mat getX_D_neg_1_2(const arma::uvec &Y, const arma::mat &X, const arma::uvec &sorted_unique_labels,
                         std::list<arma::uvec> &simlist, std::list<arma::uvec> &difflist){

  // Compute X_D
  arma::mat X_D = arma::zeros(X.n_cols, X.n_cols);

  std::list<arma::uvec>::iterator simit = simlist.begin();
  std::list<arma::uvec>::iterator simend = simlist.end();
  std::list<arma::uvec>::iterator diffit = difflist.begin();

  for(unsigned int i = 0; i < sorted_unique_labels.n_elem && simit != simend; i++, ++simit, ++diffit){

    // Check for User Interrupt
    Rcpp::checkUserInterrupt();

    // Matrix of points with current label
    arma::uvec simindices = *simit;
    arma::uvec diffindices = *diffit;

    // Fill X_D
    for(unsigned int j = 0; j < simindices.n_elem; j++){

      if(j % 50)
        Rcpp::checkUserInterrupt();

      arma::rowvec point = X.row(simindices[j]);

      for(unsigned int k = 0; k < diffindices.n_elem; k++){

        arma::rowvec diff = point - X.row(diffindices[k]);
        X_D = X_D + diff.t() * diff;

      }

    }

  }

  arma::mat X_D_neg_1_2 = NegSqrtMat(X_D);

  return X_D_neg_1_2;

}



/**
 *
 * A function to approximate X_D_neg_1_2.
 *
 */
// [[Rcpp::export]]
arma::mat Approximate_X_D_neg_1_2(const arma::uvec &Y, const arma::mat &X){

  arma::mat X_D = arma::zeros(X.n_cols, X.n_cols);
  arma::uvec sorted_unique_labels = arma::sort(arma::unique(Y));
  arma::mat means = getMeans(Y, X, sorted_unique_labels);

  for(unsigned int i = 0; i < sorted_unique_labels.n_elem - 1; i++){

    arma::uvec indices1 = arma::find(Y != sorted_unique_labels[i]);
    arma::mat X1 = X.rows(indices1);
    arma::rowvec mean = means.row(i);

    for(unsigned int k = 0; k < X1.n_rows; k++){

      arma::rowvec diff = X1.row(k) - mean;
      X_D += diff.t() * diff;

    }

  }

  arma::mat X_D_neg_1_2 = NegSqrtMat(X_D);

  return X_D_neg_1_2;


}



/**
 *
 * A function to approximate X_tilde_S.
 *
 */
// [[Rcpp::export]]
arma::mat Approximate_X_tilde_S(const arma::uvec &Y, const arma::mat &X, const arma::mat &X_D_neg_1_2){

  arma::mat X_tilde_S = arma::zeros(X.n_cols, X.n_cols);
  arma::uvec sorted_unique_labels = arma::sort(arma::unique(Y));
  arma::mat X1 = X * X_D_neg_1_2;
  arma::mat means = getMeans(Y, X1, sorted_unique_labels);

  for(unsigned int i = 0; i < sorted_unique_labels.n_elem; i++){

    arma::uvec indices1 = arma::find(Y == sorted_unique_labels[i]);
    arma::mat X2 = X1.rows(indices1);
    arma::rowvec mean = means.row(i);

    // Comptue X_S
    for(unsigned int j = 0; j < X2.n_rows; j++){

      arma::rowvec diff = mean - X2.row(j);
      X_tilde_S += diff.t() * diff;

    }

  }

  return X_tilde_S;

}



/**
*
* This function creates a vector of values for Mu, the
* penalty term in the optimization probelm.
*
*/

arma::vec getMuVec(arma::vec Energies){

  arma::vec XtildeS_eigvals = arma::unique(arma::sort(Energies));

  // Handle the case if there is only one eigenvalue
  if(XtildeS_eigvals.n_elem == 1){
    arma::vec MuVec = arma::linspace<arma::vec>(XtildeS_eigvals[0] - XtildeS_eigvals[0]/20, XtildeS_eigvals[0] + XtildeS_eigvals[0]/20, 40);
    return MuVec;
  }


  // Figure out where the big jumps in the eigenvalues occur
  arma::uvec slice = arma::linspace<arma::uvec>(1, XtildeS_eigvals.n_elem - 1, XtildeS_eigvals.n_elem - 1);
  arma::vec ratios = XtildeS_eigvals(slice - 1)/XtildeS_eigvals(slice);


  arma::uvec big_jumps = arma::find(ratios < 1e-03);
  unsigned int max_ind;

  if(big_jumps.n_elem != 0){

    // Take the position of the last jump
    // max_ind = big_jumps[big_jumps.n_elem - 1];
    max_ind = big_jumps[0] + 1;

  }else{

    // Take the position the biggest change
    big_jumps = arma::find(ratios == ratios.min());
    max_ind = big_jumps[0];
    // max_ind = XtildeS_eigvals.n_elem - 1;

  }


  // Take some values before the smallest eigenvalue
  arma::vec temp1 = arma::linspace<arma::vec>(XtildeS_eigvals[0]/10, XtildeS_eigvals[0] - XtildeS_eigvals[0]/10, XtildeS_eigvals.n_elem);

  // Take some values after the smallest eigenvalue and before the eigenvalue of the first jump
  arma::vec temp2 = arma::linspace<arma::vec>(XtildeS_eigvals[0] + XtildeS_eigvals[0]/10, XtildeS_eigvals[max_ind] - XtildeS_eigvals[0]/10, 3*XtildeS_eigvals.n_elem);

  // Take some values after the jump eigenvalue
  arma::vec temp3 = arma::linspace<arma::vec>(XtildeS_eigvals[max_ind] + XtildeS_eigvals[max_ind]/10, XtildeS_eigvals[max_ind]*10, XtildeS_eigvals.n_elem);

  // Take some values after the largest eigenvalue
  arma::vec temp4 = arma::linspace<arma::vec>(XtildeS_eigvals[XtildeS_eigvals.n_elem - 1] + XtildeS_eigvals[XtildeS_eigvals.n_elem - 1]/10, XtildeS_eigvals[XtildeS_eigvals.n_elem - 1]*10, XtildeS_eigvals.n_elem);

  arma::vec MuVec = XtildeS_eigvals;
  MuVec = arma::join_cols<arma::vec>(MuVec, temp1);
  MuVec = arma::join_cols<arma::vec>(MuVec, temp2);
  MuVec = arma::join_cols<arma::vec>(MuVec, temp3);
  MuVec = arma::join_cols<arma::vec>(MuVec, temp4);
  MuVec = arma::unique(MuVec);
  MuVec = arma::sort(MuVec);
  MuVec = MuVec(arma::find(MuVec > 0));

  return MuVec;

}



/**
*
* This function returns the lambdas corresponding to the Mu
* which maximizes the Fisher Information.
*
*/

Rcpp::List getBestMuLambdas(const arma::vec &XtildeS_eigvals, const arma::mat &mu_lambdas, const arma::vec &muVec, const arma::mat &FreeEnergies){
  arma::vec E  = mu_lambdas * XtildeS_eigvals;
  arma::vec E2 = mu_lambdas * arma::pow(XtildeS_eigvals, 2);
  arma::vec res = E2 - arma::pow(E,2);
  arma::vec FisherInformation = (1.0/(arma::pow(muVec,2))) % (E2 - arma::pow(E,2));
  unsigned int best_ind = FisherInformation.index_max();
  arma::vec Lambdas = arma::conv_to<arma::vec>::from(mu_lambdas.row(best_ind));
  return Rcpp::List::create(Rcpp::Named("Lambdas") = Lambdas,
                            Rcpp::Named("Best_Mu_index") = best_ind,
                            Rcpp::Named("Best_Mu") = muVec[best_ind],
                            Rcpp::Named("mu_lambdas") = mu_lambdas,
                            Rcpp::Named("FisherInformation") = FisherInformation,
                            Rcpp::Named("FreeEnergies") = FreeEnergies);
}



/**
*
* This function computes the eigenvalues (i.e lambdas) for
* the optimal matrix S (the matrix which solves the original
* optimization problem) using the supplied values of Mu in
* MuVec.
*
*/

Rcpp::List genLambdas(const arma::vec &XtildeS_eigvals, const arma::vec &MuVec){

  // For each value of Mu, compute the eigenvalues which the
  // corresponding eigenvector is associated with.
  arma::mat mu_lambdas = arma::zeros(MuVec.n_elem, XtildeS_eigvals.n_elem);
  arma::mat FreeEnergies = arma::zeros(MuVec.n_elem, XtildeS_eigvals.n_elem);

  for(unsigned int i = 0; i < mu_lambdas.n_rows; i++){
    for(unsigned int j = 0; j < mu_lambdas.n_cols; j++){
      FreeEnergies(i,j) = exp((-1/MuVec[i])*XtildeS_eigvals[j]);
      mu_lambdas(i,j) = 1.0/arma::sum(arma::exp((-1/MuVec[i])*(XtildeS_eigvals - XtildeS_eigvals[j])));
    }
  }

  Rcpp::List lst = getBestMuLambdas(XtildeS_eigvals, mu_lambdas, MuVec, FreeEnergies);
  return lst;
}



/**
*
* This function computes S_1_2.
*
*/
// [[Rcpp::export]]
Rcpp::List getS_1_2(const arma::uvec &Y, const arma::mat &X, const arma::mat &X_D_neg_1_2,
                   const arma::uvec &sorted_unique_labels, std::list<arma::uvec> &simlist){

  ////////////////////////////////////////////
  // First compute XtildeS
  ////////////////////////////////////////////

  arma::mat X_tilde_S = arma::zeros(X.n_cols, X.n_cols);;
  std::list<arma::uvec>::iterator simit = simlist.begin();
  std::list<arma::uvec>::iterator simend = simlist.end();
  arma::mat X2 = X * X_D_neg_1_2;

  for(unsigned int i = 0; i < sorted_unique_labels.n_elem && simit != simend; i++, ++simit){

    // Matrix of points with current label
    arma::uvec simindices = *simit;

    for(unsigned int j = 0; j < (simindices.n_elem - 1); j++){

      arma::rowvec point = X2.row(simindices[j]);

      for(unsigned int k = j + 1; k < simindices.n_elem; k++){
        arma::rowvec diff = point - X2.row(simindices[k])  ;
        X_tilde_S = X_tilde_S + diff.t() * diff;

      }

    }

  }

  // Get the eigenvalues and eigenvectors of XtildeS
  arma::vec XtildeS_eigvals;
  arma::mat XtildeS_eigvecs;
  arma::mat XtildeS_eigvecs2;
  arma::svd(XtildeS_eigvecs, XtildeS_eigvals, XtildeS_eigvecs2, X_tilde_S);

  // Compute Lambdas
  arma::vec MuVec = getMuVec(XtildeS_eigvals);
  Rcpp::List lst = genLambdas(XtildeS_eigvals, MuVec);
  arma::vec Lambdas = lst["Lambdas"];

  // Take the square root of the lambdas when they are > 0
  for(unsigned int i = 0; i < Lambdas.n_elem; i++)
    if(Lambdas[i] > 0)
      Lambdas[i] = sqrt(Lambdas[i]);
    else
      Lambdas[i] = 0;

  arma::mat S_1_2 = XtildeS_eigvecs * arma::diagmat(Lambdas) * XtildeS_eigvecs.t();

  return Rcpp::List::create(Rcpp::Named("S_1_2") = S_1_2,
                            Rcpp::Named("S_eigvecs") = XtildeS_eigvecs,
                            Rcpp::Named("Lambdas") = Lambdas,
                            Rcpp::Named("Energies") = XtildeS_eigvals,
                            Rcpp::Named("MuVec") = MuVec,
                            Rcpp::Named("Best_Mu_index") = lst["Best_Mu_index"],
                            Rcpp::Named("Best_Mu") = lst["Best_Mu"],
                            Rcpp::Named("mu_lambdas") = lst["mu_lambdas"],
                            Rcpp::Named("FisherInformation") = lst["FisherInformation"],
                            Rcpp::Named("FreeEnergies") = lst["FreeEnergies"]);

}



/**
*
* This function computes the global FENN method.
*
*/
// [[Rcpp::export]]
Rcpp::List fenn(const arma::uvec &Y, const arma::mat &X, const arma::uvec &sorted_unique_labels){

  // For each label, find the indices with similar and different
  // class labels in Y
  std::list<arma::uvec> simlist;
  std::list<arma::uvec> difflist;

  for(unsigned int i = 0; i < sorted_unique_labels.n_elem; i++){

    arma::uvec indices1 = arma::find(Y == sorted_unique_labels[i]);
    simlist.push_back(indices1);
    arma::uvec indices2 = arma::find(Y > sorted_unique_labels[i]);
    difflist.push_back(indices2);

  }

  // First compute X_D_neg_1_2
  arma::mat X_D_neg_1_2 = getX_D_neg_1_2(Y, X, sorted_unique_labels, simlist, difflist);

  // Compute S_neg_1_2
  Rcpp::List lst = getS_1_2(Y, X, X_D_neg_1_2, sorted_unique_labels, simlist);
  arma::mat S_1_2 = lst["S_1_2"];
  arma::mat FENNTransform = S_1_2 * X_D_neg_1_2;

  return Rcpp::List::create(Rcpp::Named("X_D_neg_1_2") = X_D_neg_1_2,
                            Rcpp::Named("FENNTransform") = S_1_2 * X_D_neg_1_2,
                            Rcpp::Named("Lambdas") = lst["Lambdas"],
                            Rcpp::Named("SimilarityDirections") = lst["S_eigvecs"],
                            Rcpp::Named("Energies") = lst["Energies"],
                            Rcpp::Named("MuVec") = lst["MuVec"],
                            Rcpp::Named("FisherInformation") = lst["FisherInformation"],
                            Rcpp::Named("FreeEnergies") = lst["FreeEnergies"],
                            Rcpp::Named("MuLambdas") = lst["mu_lambdas"]);

}


