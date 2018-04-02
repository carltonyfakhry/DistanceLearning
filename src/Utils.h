#ifndef UTILS
#define UTILS

#include <iostream>
#include <algorithm>
#include <cmath>
#include <Rmath.h>
#include <RcppArmadillo.h>
#include <exception>
#include <limits>

// [[Rcpp::depends(RcppArmadillo)]]



/**
 *
 *  Count the number of distinct pairs of points with different
 *  class labels, without taking order into account.
 *
 */

unsigned int getPairsCount(const arma::uvec &testY, const arma::uvec &labels);



/**
 *
 * This function returns indices for points in the local neighborhood
 * (euclidean distance) of a given focal point and their respective
 * distances.
 *
 */

Rcpp::List get_neighborhood(arma::mat data, arma::rowvec point, unsigned int k);



/**
 *
 * This function returns indices for points in the local neighborhood
 * (euclidean distance) of a given focal point.
 *
 */

arma::uvec get_neighborhood2(const arma::mat &data, const arma::rowvec &point, unsigned int k);



/**
 *
 * This function computes the pseudo inverse of a square matrix.
 *
 */

arma::mat getPseudoInverse(const arma::mat &X, double tolerance = std::sqrt(std::numeric_limits<double>::epsilon()));



/**
 *
 * This function computes the inverse of a square matrix.
 *
 */

arma::mat NegSqrtMat(const arma::mat &X, double tolerance = std::sqrt(std::numeric_limits<double>::epsilon()));



/**
 *
 * This function computes the square root of a square
 * matrix.
 *
 */

arma::mat SqrtMat(const arma::mat &X, double tolerance = std::sqrt(std::numeric_limits<double>::epsilon()));



/**
 *
 * This function computes the means of points in each class label.
 *
 */

arma::mat getMeans(const arma::uvec &Y, const arma::mat &X, const arma::uvec &labels);



/**
 *
 * Predict labels using K-NN.
 *
 */

int KNN(const arma::uvec &Y, const arma::uvec &labels);



#endif
