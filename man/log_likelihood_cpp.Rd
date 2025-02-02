% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{log_likelihood_cpp}
\alias{log_likelihood_cpp}
\title{Log-Likelihood for Binomial Logistic Regression}
\usage{
log_likelihood_cpp(X, y, beta)
}
\arguments{
\item{X}{A numeric matrix of predictors.}

\item{y}{A numeric vector of binary outcomes (0 or 1).}

\item{beta}{A numeric vector of regression coefficients.}
}
\value{
The negative log-likelihood value.
}
\description{
This function computes the log-likelihood for logistic regression using the binomial distribution.
}
