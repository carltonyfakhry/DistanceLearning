#include "Utils.h"


// This code was adapted from the author's original code
// http://www.cs.cmu.edu/%7Eepxing/papers/Old_papers/code_Metric_online.tar.gz


/**
 *
 * References:
 *
 * Eric P. Xing, Michael I. Jordan, Stuart J Russell, and Andrew Y. Ng. Distance Met-
 * ric Learning with Application to Clustering with Side-Information. In S. Becker,
 * S. Thrun, and K. Obermayer, editors, Advances in Neural Information Processing
 * Systems 15, pages 521–528. MIT Press, 2003.
 *
 */



/**
 *
 * This functions takes the sum of all the outer products
 * of vectors in a given matrix.
 *
 */
// [[Rcpp::export]]
arma::mat sumOuterProducts(const arma::mat &X, const arma::mat &S, const int n_cols){

  arma::mat OuterProds = arma::zeros(n_cols, n_cols);

  // OuterProds is the sum of all products x.x^T
  for(unsigned int i = 0; i < S.n_rows; i++){
    arma::rowvec diff = X.row(S(i,0)) - X.row(S(i,1));
    OuterProds = OuterProds + diff.t() * diff;
  }

  return OuterProds;

}



/**
 *
 * This function returns the disimilarity matrix i.e an N x 2 matrix
 * where each row is a pair of indices for two data points which belong
 * to different classes.
 *
 */
// [[Rcpp::export]]
arma::umat getD(const arma::uvec &Y, const arma::mat &X, const arma::uvec &labels){

  // Get count for unique pairs of points with different
  // labels
  int pairs_count = getPairsCount(Y, labels);

  // Initialize D
  arma::umat D = arma::zeros<arma::umat>(pairs_count, 2);

  // Fill matrix XdotD
  unsigned int rownum = 0;
  for(unsigned int i = 0; i < labels.n_elem; i++){

    // Check for User Interrupt
    Rcpp::checkUserInterrupt();

    // Indices of points with current label
    arma::uvec group = arma::find(Y == labels[i]);

    // Indices of points with different label
    arma::uvec diffgroup = arma::find(Y > labels[i]);

    // Fill D
    for(unsigned int j = 0; j < group.n_elem; j++){
      for(unsigned int k = 0; k < diffgroup.n_elem; k++){
        D(rownum, 0) = group[j];
        D(rownum, 1) = diffgroup[k];
        rownum++;
      }
    }

  }

  return D;

}



/**
 *
 * This function computes X_S. The rows of X_S consist
 * of (x-y).(x-y)^T for pairs of points x, y with the same
 * class labels. XdotS computed with getXdotS, contains
 * rows with (x-y) which can be used to compute X_S.
 *
 */
// [[Rcpp::export]]
arma::mat getX_S(const arma::uvec &Y, const arma::mat &X, const arma::uvec &labels){

  // X_S is the sum of all products (x-y).(x-y)^T
  arma::mat X_S = arma::zeros(X.n_cols, X.n_cols);

  for(unsigned int i = 0; i < labels.n_elem; i++){

    // Matrix of points with current label
    arma::uvec group = arma::find(Y == labels[i]);

    for(unsigned int j = 0; j < group.n_elem - 1; j++){

      Rcpp::checkUserInterrupt();

      for(unsigned int k = j + 1; k < group.n_elem; k++){

        arma::rowvec diff = X.row(group[j]) - X.row(group[k]);
        X_S = X_S + diff.t() * diff;

      }

    }

  }

  return X_S;

}



/**
 *
 * This functions turns an n by n matrix to a vector
 * of length n^2.
 *
 */

arma::vec Mat2Vec(const arma::mat &X){

  arma::vec vecrep(X.n_rows * X.n_cols);
  int index = 0;

  for(unsigned int i = 0; i < X.n_cols; i++){

    for(unsigned int j = 0; j < X.n_rows; j++){

      vecrep[index] = X(j,i);
      index++;

    }

  }

  return vecrep;

}



/**
 *
 * This functions turns a vector of length n^2 to an
 * n by n matrix.
 *
 */

arma::mat Vec2Mat(const arma::vec &vecrep, const arma::mat &X){

  arma::mat matrep = arma::zeros(X.n_rows, X.n_cols);
  int index = 0;

  for(unsigned int i = 0; i < matrep.n_cols; i++){

    for(unsigned int j = 0; j < matrep.n_rows; j++){

      matrep(j,i) = vecrep[index];
      index++;

    }

  }

  return matrep;

}



/**
 *
 * This function projects the Mahalanobis matrix back on to the set of
 * semi-positive definite matrices or C_2 as described in [1].
 *
 */

arma::mat ProjectToConstraint2(const arma::mat &A){

  arma::mat projectedA;
  arma::vec s;
  arma::mat U;
  arma::mat V;

  // Attempt to do SVD, and if SVD fails then stop
  bool success = arma::svd(U, s, V, A);

  if(!success){

    success = arma::svd(U, s, V, A, "std");

    if(!success){

      Rcpp::stop("SVD failed. Unable to project to Constraint 2!");

    }

  }

  // If SVD succeeds then project onto C_2
  arma::uvec indices = arma::find(s < 0);
  s.rows(indices).fill(0);
  arma::mat Sigma = arma::diagmat(s);
  projectedA = U * Sigma * V.t();

  return projectedA;

}



/**
 *
 * This function computes the gradient of the objective function
 * in equation (6) in [1]. The gradient is chosen such that it is
 * perpendicular to constraint set C_1.
 *
 */

arma::mat getObjectiveGradient(const arma::mat &A,  const arma::mat &X,
                               const arma::umat &D, const arma::mat &Grad2){

  double fudge = 0.000001;
  arma::mat Grad1 = arma::zeros(Grad2.n_cols, Grad2.n_cols);

  // Compute the gradient of the objective i.e \sum_i (x_i * x_i^T)/(tr(x_i^T * A * x_i))^1/2
  for(unsigned int i = 0; i < D.n_rows; i++){
    arma::rowvec diff = X.row(D(i,0)) - X.row(D(i,1));
    arma::mat X_tau = diff.t() * diff;
    double denom = sqrt(arma::trace(A * (X_tau + fudge)));
    if(denom > 0){
      Grad1 = Grad1 + 0.5 * X_tau / denom ;
    }
  }

  // Project the gradient perpendicularly towards the constraint set C_1
  arma::vec vecGrad1 = Mat2Vec(Grad1);
  arma::vec vecGrad2 = Mat2Vec(Grad2);

  vecGrad2 = vecGrad2 / arma::norm(vecGrad2, 2);
  arma::vec vecProjGrad = vecGrad1 - arma::dot(vecGrad2, vecGrad1) * vecGrad2;
  vecProjGrad = vecProjGrad / arma::norm(vecProjGrad, 2);
  arma::mat ProjGrad = Vec2Mat(vecProjGrad, Grad1);

  return ProjGrad;

}



/**
 *
 * Compute the objective function in equation (6) in [1].
 *
 */

double getObjective(const arma::mat &A, const arma::mat &X, const arma::umat &D){

  arma::mat obj(1,1);

  // Compute the objective function i.e \sum_i sqrt(x_i^T * A * x_i)
  for(unsigned int i = 0; i < D.n_rows; i++){

    arma::rowvec diff = X.row(D(i,0)) - X.row(D(i,1));
    obj = obj + arma::sqrt(diff * A * diff.t());

  }

  return obj(0,0);

}



/**
 *
 * This function learns the Mahalanobis matrix as desribed in the
 * algorithm in Figure 1 in [1].
 *
 */

Rcpp::List learnxing(const arma::uvec &Y, const arma::mat &X,
                     const arma::mat &X_S, const arma::umat &D, const arma::vec &w,
                     int t, double learning_rate, double epsilon, double error,
                     int max_iterations){


  double fudge = 0.000001;

  // Make w is a unit vector
  double w_len = arma::norm(w, 2);
  arma::vec w_1;
  if(w_len > 0){

    w_1 = w / arma::norm(w, 2);

  }else{

    Rcpp::stop("weight_matrix has only 0 entries, cannot compute the metric!");

  }

  // Distance from origin to w^T * x = t plane
  double t_1 = t / w_len;

  // Initialize Mahalanobis matrix
  arma::mat A = arma::eye(X.n_cols, X.n_cols);

  // Compute initial objective function
  double objective = getObjective(A, X, D);
  double objective_previous = 0;

  // Gradient of the first constraint C_1 is constant
  // throughout being equal to X_S
  arma::mat Grad2 = X_S;

  arma::mat prev_A = A;
  arma::mat prev_A2;
  arma::mat M = getObjectiveGradient(A, X, D, Grad2);

  int iter1 = 0;
  bool converged = false;
  double delta;

  while(true){

    int iter2 = 0;
    bool converged2 = false;

    while(iter2 < max_iterations && !converged2){

      prev_A2 = A;

      // Project onto C_1
      // C_1 : \sum_{i,j \in S} d_ij' A d_ij <= t              (1)
      // (1) can be rewritten as a linear constraint: w^T x = t,
      // where x is the unrolled matrix of A,
      // w is also an unrolled matrix of weight_matrix where
      // weight_matrix = W_{kl}= \sum_{i,j \in S} d_ij^k * (d_ij^T)^l

      arma::vec x = Mat2Vec(prev_A2);
      double val = arma::dot(w, x);

      if(val <= t){

        A = prev_A2;

      }else{

        x = x + (t_1 - arma::dot(w_1, x)) * w_1;
        A = Vec2Mat(x, prev_A2);

      }

      // Enforce A to be symmetric
      A = (A + A.t())/2;

      // Project on C_2, the set of positive semi-definite
      // matrices
      A = ProjectToConstraint2(A);

      x = Mat2Vec(A);
      double constraint_function_value = arma::dot(w, x);
      iter2++;
      double error2 = (constraint_function_value - t)/t;

      if(error2 < epsilon){

        converged2 = true;

      }

    }

    objective_previous = objective;
    double objective = getObjective(A, X, D);

    if(((objective > objective_previous) | (iter1 == 1)) && converged2){

      learning_rate = learning_rate * 1.05;
      prev_A = A;
      M = getObjectiveGradient(A, X, D, Grad2);
      A = A + learning_rate * M;

    }else{

      learning_rate = learning_rate/2;
      A = prev_A + learning_rate * M;

    }

    delta = arma::norm(learning_rate * M, "fro")/ arma::norm(prev_A + fudge, "fro");
    iter1++;

    if(iter1 == max_iterations || delta <= epsilon){
      break;
    }

  }

  if(delta <= epsilon){

    converged = true;

  }

  A = prev_A;

  return Rcpp::List::create(Rcpp::Named("converged") = converged,
                            Rcpp::Named("A") = A);

}



/**
 *
 * This function predicts the class label on a new set of data after
 * the data is transformed using the learned Mahalanobis matrix.
 *
 */
// [[Rcpp::export]]
Rcpp::List xing(const arma::uvec &Y, const arma::mat &X,
                const arma::uvec &labels, const arma::mat &X_S,
                const arma::umat &D, const arma::mat &weight_matrix,
                double learning_rate, double error, double epsilon,
                int max_iterations){

  int t = 1;

  // Compute w, a weighting vector which is a function of the similarity matrix
  // which is to be used when projecting onto the first constraint C_1
  arma::vec w = Mat2Vec(weight_matrix);

  // Learn the Mahalanobis matrix
  Rcpp::List lst = learnxing(Y, X, X_S, D, w, t, learning_rate, epsilon, error, max_iterations);
  arma::mat A = lst["A"];
  bool converged = lst["converged"];

  // Transform data according to the metric A
  arma::mat A_1_2 = SqrtMat(A);
  arma::mat transformedX = X * A_1_2;

  if(!converged){
    Rcpp::warning("Metric solution did not converge with the given input parameters. Results are not based on the optimal metric!");
  }

  return Rcpp::List::create(Rcpp::Named("XingTransform") = A_1_2,
                            Rcpp::Named("transformedX") = transformedX);

}
