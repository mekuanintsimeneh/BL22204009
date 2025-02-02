% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/code.R
\name{gibbs_exponential_R}
\alias{gibbs_exponential_R}
\title{Gibbs Sampling for Exponential Distribution}
\usage{
gibbs_exponential_R(y, alpha, beta, n_iter = 1000, return_mean = FALSE)
}
\arguments{
\item{y}{Numeric vector of observed data.}

\item{alpha}{Prior shape parameter for the Gamma distribution.}

\item{beta}{Prior rate parameter for the Gamma distribution.}

\item{n_iter}{Number of Gibbs sampling iterations.}

\item{return_mean}{Logical. If TRUE, returns the posterior mean instead of samples.}
}
\value{
A vector of sampled lambda values or the posterior mean if return_mean = TRUE.
}
\description{
Performs Gibbs sampling for the rate parameter (lambda) of an exponential distribution.
}
\examples{
set.seed(123)
y <- rexp(100, rate = 2)  # Generate data with true lambda = 2
samples <- gibbs_exponential_R(y, alpha = 1, beta = 1, n_iter = 1000)
hist(samples, breaks = 30, main = "Posterior Samples of Lambda")
mean_lambda <- gibbs_exponential_R(y, alpha = 1, beta = 1, n_iter = 1000, return_mean = TRUE)
print(mean_lambda)
}
