% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/code.R
\name{log_likelihood_R}
\alias{log_likelihood_R}
\title{Log-Likelihood for Binomial Logistic Regression}
\usage{
log_likelihood_R(X, y, beta)
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
\examples{
X <- matrix(rnorm(10 * 2), ncol = 2)
beta <- c(0.5, -1)
y <- rbinom(10, 1, prob = logistic_R(X \%*\% beta))
log_likelihood_R(X, y, beta)
}
