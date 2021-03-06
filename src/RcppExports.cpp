// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// adamenn
arma::ivec adamenn(const arma::uvec& Y, const arma::mat& X, const arma::mat& newX, const arma::uvec& labels, int K, int K_0, int K_1, int K_2, int L, double c, int n_iterations);
RcppExport SEXP _DistanceLearning_adamenn(SEXP YSEXP, SEXP XSEXP, SEXP newXSEXP, SEXP labelsSEXP, SEXP KSEXP, SEXP K_0SEXP, SEXP K_1SEXP, SEXP K_2SEXP, SEXP LSEXP, SEXP cSEXP, SEXP n_iterationsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type newX(newXSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type labels(labelsSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type K_0(K_0SEXP);
    Rcpp::traits::input_parameter< int >::type K_1(K_1SEXP);
    Rcpp::traits::input_parameter< int >::type K_2(K_2SEXP);
    Rcpp::traits::input_parameter< int >::type L(LSEXP);
    Rcpp::traits::input_parameter< double >::type c(cSEXP);
    Rcpp::traits::input_parameter< int >::type n_iterations(n_iterationsSEXP);
    rcpp_result_gen = Rcpp::wrap(adamenn(Y, X, newX, labels, K, K_0, K_1, K_2, L, c, n_iterations));
    return rcpp_result_gen;
END_RCPP
}
// dann
arma::ivec dann(arma::uvec& Y, arma::mat& X, arma::mat& newX, int K, int K_M, double epsilon, int n_iterations);
RcppExport SEXP _DistanceLearning_dann(SEXP YSEXP, SEXP XSEXP, SEXP newXSEXP, SEXP KSEXP, SEXP K_MSEXP, SEXP epsilonSEXP, SEXP n_iterationsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type newX(newXSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type K_M(K_MSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< int >::type n_iterations(n_iterationsSEXP);
    rcpp_result_gen = Rcpp::wrap(dann(Y, X, newX, K, K_M, epsilon, n_iterations));
    return rcpp_result_gen;
END_RCPP
}
// dca
Rcpp::List dca(const arma::mat& X, const arma::ivec& chunklets, const arma::uvec& unique_chunklets, const arma::umat& D, int total_dims);
RcppExport SEXP _DistanceLearning_dca(SEXP XSEXP, SEXP chunkletsSEXP, SEXP unique_chunkletsSEXP, SEXP DSEXP, SEXP total_dimsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::ivec& >::type chunklets(chunkletsSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type unique_chunklets(unique_chunkletsSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type D(DSEXP);
    Rcpp::traits::input_parameter< int >::type total_dims(total_dimsSEXP);
    rcpp_result_gen = Rcpp::wrap(dca(X, chunklets, unique_chunklets, D, total_dims));
    return rcpp_result_gen;
END_RCPP
}
// getX_D_neg_1_2
arma::mat getX_D_neg_1_2(const arma::uvec& Y, const arma::mat& X, const arma::uvec& sorted_unique_labels, std::list<arma::uvec>& simlist, std::list<arma::uvec>& difflist);
RcppExport SEXP _DistanceLearning_getX_D_neg_1_2(SEXP YSEXP, SEXP XSEXP, SEXP sorted_unique_labelsSEXP, SEXP simlistSEXP, SEXP difflistSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type sorted_unique_labels(sorted_unique_labelsSEXP);
    Rcpp::traits::input_parameter< std::list<arma::uvec>& >::type simlist(simlistSEXP);
    Rcpp::traits::input_parameter< std::list<arma::uvec>& >::type difflist(difflistSEXP);
    rcpp_result_gen = Rcpp::wrap(getX_D_neg_1_2(Y, X, sorted_unique_labels, simlist, difflist));
    return rcpp_result_gen;
END_RCPP
}
// Approximate_X_D_neg_1_2
arma::mat Approximate_X_D_neg_1_2(const arma::uvec& Y, const arma::mat& X);
RcppExport SEXP _DistanceLearning_Approximate_X_D_neg_1_2(SEXP YSEXP, SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(Approximate_X_D_neg_1_2(Y, X));
    return rcpp_result_gen;
END_RCPP
}
// Approximate_X_tilde_S
arma::mat Approximate_X_tilde_S(const arma::uvec& Y, const arma::mat& X, const arma::mat& X_D_neg_1_2);
RcppExport SEXP _DistanceLearning_Approximate_X_tilde_S(SEXP YSEXP, SEXP XSEXP, SEXP X_D_neg_1_2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_D_neg_1_2(X_D_neg_1_2SEXP);
    rcpp_result_gen = Rcpp::wrap(Approximate_X_tilde_S(Y, X, X_D_neg_1_2));
    return rcpp_result_gen;
END_RCPP
}
// getS_1_2
Rcpp::List getS_1_2(const arma::uvec& Y, const arma::mat& X, const arma::mat& X_D_neg_1_2, const arma::uvec& sorted_unique_labels, std::list<arma::uvec>& simlist);
RcppExport SEXP _DistanceLearning_getS_1_2(SEXP YSEXP, SEXP XSEXP, SEXP X_D_neg_1_2SEXP, SEXP sorted_unique_labelsSEXP, SEXP simlistSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_D_neg_1_2(X_D_neg_1_2SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type sorted_unique_labels(sorted_unique_labelsSEXP);
    Rcpp::traits::input_parameter< std::list<arma::uvec>& >::type simlist(simlistSEXP);
    rcpp_result_gen = Rcpp::wrap(getS_1_2(Y, X, X_D_neg_1_2, sorted_unique_labels, simlist));
    return rcpp_result_gen;
END_RCPP
}
// fenn
Rcpp::List fenn(const arma::uvec& Y, const arma::mat& X, const arma::uvec& sorted_unique_labels);
RcppExport SEXP _DistanceLearning_fenn(SEXP YSEXP, SEXP XSEXP, SEXP sorted_unique_labelsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type sorted_unique_labels(sorted_unique_labelsSEXP);
    rcpp_result_gen = Rcpp::wrap(fenn(Y, X, sorted_unique_labels));
    return rcpp_result_gen;
END_RCPP
}
// lfda
Rcpp::List lfda(const arma::uvec& Y, const arma::mat& X, const arma::uvec& labels, int total_dims, std::string method);
RcppExport SEXP _DistanceLearning_lfda(SEXP YSEXP, SEXP XSEXP, SEXP labelsSEXP, SEXP total_dimsSEXP, SEXP methodSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type labels(labelsSEXP);
    Rcpp::traits::input_parameter< int >::type total_dims(total_dimsSEXP);
    Rcpp::traits::input_parameter< std::string >::type method(methodSEXP);
    rcpp_result_gen = Rcpp::wrap(lfda(Y, X, labels, total_dims, method));
    return rcpp_result_gen;
END_RCPP
}
// nca
Rcpp::List nca(const arma::uvec& Y, const arma::mat& X, const arma::uvec& labels, unsigned int iterations, double learning_rate);
RcppExport SEXP _DistanceLearning_nca(SEXP YSEXP, SEXP XSEXP, SEXP labelsSEXP, SEXP iterationsSEXP, SEXP learning_rateSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type labels(labelsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type learning_rate(learning_rateSEXP);
    rcpp_result_gen = Rcpp::wrap(nca(Y, X, labels, iterations, learning_rate));
    return rcpp_result_gen;
END_RCPP
}
// rca
Rcpp::List rca(const arma::mat& X, const arma::ivec& chunklets, const arma::uvec& unique_chunklets, int total_dims);
RcppExport SEXP _DistanceLearning_rca(SEXP XSEXP, SEXP chunkletsSEXP, SEXP unique_chunkletsSEXP, SEXP total_dimsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::ivec& >::type chunklets(chunkletsSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type unique_chunklets(unique_chunkletsSEXP);
    Rcpp::traits::input_parameter< int >::type total_dims(total_dimsSEXP);
    rcpp_result_gen = Rcpp::wrap(rca(X, chunklets, unique_chunklets, total_dims));
    return rcpp_result_gen;
END_RCPP
}
// sumOuterProducts
arma::mat sumOuterProducts(const arma::mat& X, const arma::mat& S, const int n_cols);
RcppExport SEXP _DistanceLearning_sumOuterProducts(SEXP XSEXP, SEXP SSEXP, SEXP n_colsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const int >::type n_cols(n_colsSEXP);
    rcpp_result_gen = Rcpp::wrap(sumOuterProducts(X, S, n_cols));
    return rcpp_result_gen;
END_RCPP
}
// getD
arma::umat getD(const arma::uvec& Y, const arma::mat& X, const arma::uvec& labels);
RcppExport SEXP _DistanceLearning_getD(SEXP YSEXP, SEXP XSEXP, SEXP labelsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type labels(labelsSEXP);
    rcpp_result_gen = Rcpp::wrap(getD(Y, X, labels));
    return rcpp_result_gen;
END_RCPP
}
// getX_S
arma::mat getX_S(const arma::uvec& Y, const arma::mat& X, const arma::uvec& labels);
RcppExport SEXP _DistanceLearning_getX_S(SEXP YSEXP, SEXP XSEXP, SEXP labelsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type labels(labelsSEXP);
    rcpp_result_gen = Rcpp::wrap(getX_S(Y, X, labels));
    return rcpp_result_gen;
END_RCPP
}
// xing
Rcpp::List xing(const arma::uvec& Y, const arma::mat& X, const arma::uvec& labels, const arma::mat& X_S, const arma::umat& D, const arma::mat& weight_matrix, double learning_rate, double error, double epsilon, int max_iterations);
RcppExport SEXP _DistanceLearning_xing(SEXP YSEXP, SEXP XSEXP, SEXP labelsSEXP, SEXP X_SSEXP, SEXP DSEXP, SEXP weight_matrixSEXP, SEXP learning_rateSEXP, SEXP errorSEXP, SEXP epsilonSEXP, SEXP max_iterationsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type labels(labelsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_S(X_SSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type D(DSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type weight_matrix(weight_matrixSEXP);
    Rcpp::traits::input_parameter< double >::type learning_rate(learning_rateSEXP);
    Rcpp::traits::input_parameter< double >::type error(errorSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< int >::type max_iterations(max_iterationsSEXP);
    rcpp_result_gen = Rcpp::wrap(xing(Y, X, labels, X_S, D, weight_matrix, learning_rate, error, epsilon, max_iterations));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_DistanceLearning_adamenn", (DL_FUNC) &_DistanceLearning_adamenn, 11},
    {"_DistanceLearning_dann", (DL_FUNC) &_DistanceLearning_dann, 7},
    {"_DistanceLearning_dca", (DL_FUNC) &_DistanceLearning_dca, 5},
    {"_DistanceLearning_getX_D_neg_1_2", (DL_FUNC) &_DistanceLearning_getX_D_neg_1_2, 5},
    {"_DistanceLearning_Approximate_X_D_neg_1_2", (DL_FUNC) &_DistanceLearning_Approximate_X_D_neg_1_2, 2},
    {"_DistanceLearning_Approximate_X_tilde_S", (DL_FUNC) &_DistanceLearning_Approximate_X_tilde_S, 3},
    {"_DistanceLearning_getS_1_2", (DL_FUNC) &_DistanceLearning_getS_1_2, 5},
    {"_DistanceLearning_fenn", (DL_FUNC) &_DistanceLearning_fenn, 3},
    {"_DistanceLearning_lfda", (DL_FUNC) &_DistanceLearning_lfda, 5},
    {"_DistanceLearning_nca", (DL_FUNC) &_DistanceLearning_nca, 5},
    {"_DistanceLearning_rca", (DL_FUNC) &_DistanceLearning_rca, 4},
    {"_DistanceLearning_sumOuterProducts", (DL_FUNC) &_DistanceLearning_sumOuterProducts, 3},
    {"_DistanceLearning_getD", (DL_FUNC) &_DistanceLearning_getD, 3},
    {"_DistanceLearning_getX_S", (DL_FUNC) &_DistanceLearning_getX_S, 3},
    {"_DistanceLearning_xing", (DL_FUNC) &_DistanceLearning_xing, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_DistanceLearning(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
