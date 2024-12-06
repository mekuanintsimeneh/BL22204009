
#' Gibbs Sampling for Exponential Distribution
#'
#' Performs Gibbs sampling for the rate parameter (lambda) of an exponential distribution.
#' @import MASS
#' @import boot
#' @import bootstrap
#' @import DAAG
#' @import coda
#' @importFrom ggplot2 ggplot
#' @importFrom stats rgamma
#' @param y Numeric vector of observed data.
#' @param alpha Prior shape parameter for the Gamma distribution.
#' @param beta Prior rate parameter for the Gamma distribution.
#' @param n_iter Number of Gibbs sampling iterations.
#' @param return_mean Logical. If TRUE, returns the posterior mean instead of samples.
#' @return A vector of sampled lambda values or the posterior mean if return_mean = TRUE.
#' @examples
#' set.seed(123)
#' y <- rexp(100, rate = 2)  # Generate data with true lambda = 2
#' samples <- gibbs_exponential_R(y, alpha = 1, beta = 1, n_iter = 1000)
#' hist(samples, breaks = 30, main = "Posterior Samples of Lambda")
#' mean_lambda <- gibbs_exponential_R(y, alpha = 1, beta = 1, n_iter = 1000, return_mean = TRUE)
#' print(mean_lambda)
#' @export
gibbs_exponential_R <- function(y, alpha, beta, n_iter = 1000, return_mean = FALSE) {
  n <- length(y)             # Number of observations
  sum_y <- sum(y)            # Sufficient statistic for exponential data
  lambda_samples <- numeric(n_iter) # Vector to store samples

  for (iter in 1:n_iter) {
    # Posterior parameters for Gamma distribution
    alpha_post <- alpha + n
    beta_post <- beta + sum_y

    # Sample lambda from the posterior Gamma(alpha_post, beta_post)
    lambda_samples[iter] <- rgamma(1, shape = alpha_post, rate = beta_post)
  }

  if (return_mean) {
    return(mean(lambda_samples)) # Return the mean of the sampled lambdas
  } else {
    return(lambda_samples) # Return all samples
  }
}

#' Gibbs Sampling for Poisson-Gamma Model
#'
#' Performs Gibbs sampling for the rate parameter (lambda) of a Poisson likelihood with a Gamma prior.
#' The model assumes that the data follow a Poisson distribution, and the prior on lambda is a Gamma distribution.
#'
#' @importFrom stats rgamma
#' @param data Numeric vector of observed counts (Poisson data).
#' @param alpha_prior Prior shape parameter for the Gamma distribution.
#' @param beta_prior Prior rate parameter for the Gamma distribution.
#' @param n_iter Number of Gibbs sampling iterations.
#' @return A numeric vector of sampled lambda values.
#' @examples
#' set.seed(123)
#' data <- c(2, 4, 1, 3, 5, 2, 4, 3)
#' samples <- bayesian_poisson_gamma_R(data, alpha_prior = 2, beta_prior = 1, n_iter = 10000)
#' hist(samples, breaks = 30, main = "Posterior Samples of Lambda (Poisson Rate)")
#' @export
bayesian_poisson_gamma_R <- function(data, alpha_prior, beta_prior, n_iter = 10000) {
  # Number of observations
  n <- length(data)

  # Sum of observed data
  sum_data <- sum(data)

  # Vector to store posterior samples
  lambda_samples <- numeric(n_iter)

  # Initial guess for lambda
  lambda_current <- 1.0

  # Gibbs sampling loop
  for (i in 1:n_iter) {
    # Posterior shape parameter: alpha + sum(data)
    posterior_shape <- alpha_prior + sum_data

    # Posterior scale parameter: beta_prior + n (number of data points)
    posterior_scale <- 1 / (1 / beta_prior + n)

    # Sample lambda from the posterior Gamma distribution
    lambda_current <- rgamma(1, shape = posterior_shape, rate = posterior_scale)

    # Store the sampled value of lambda
    lambda_samples[i] <- lambda_current
  }

  # Return the posterior samples of lambda
  return(lambda_samples)
}

#' Logistic function
#'
#' This is a logistic sigmoid function that maps input values to probabilities between 0 and 1.
#' @param x A numeric value or vector.
#' @return Logistic transformation of input x.
#' @return Logistic transformation of input x.
#' @export
logistic_R <- function(x) {
  return(1 / (1 + exp(-x)))
}

#' Log-Likelihood for Binomial Logistic Regression
#'
#' This function computes the log-likelihood for logistic regression using the binomial distribution.
#' @param X A numeric matrix of predictors.
#' @param y A numeric vector of binary outcomes (0 or 1).
#' @param beta A numeric vector of regression coefficients.
#' @return The negative log-likelihood value.
#' @examples
#' X <- matrix(rnorm(10 * 2), ncol = 2)
#' beta <- c(0.5, -1)
#'y <- rbinom(10, 1, prob = logistic_R(X %*% beta))
#'log_likelihood_R(X, y, beta)

#' @export
log_likelihood_R <- function(X, y, beta) {
  n <- nrow(X)
  p <- ncol(X)
  ll <- 0
  for (i in 1:n) {
    eta <- sum(X[i, ] * beta)
    pi <- logistic_R(eta)
    ll <- ll + y[i] * log(pi) + (1 - y[i]) * log(1 - pi)
  }
  return(-ll)
}

#' Gradient of the log-likelihood for Binomial Logistic Regression
#'
#' This function computes the gradient of the log-likelihood for logistic regression.
#' @param X A numeric matrix of predictors.
#' @param y A numeric vector of binary outcomes (0 or 1).
#' @param beta A numeric vector of regression coefficients.
#' @return A numeric vector of gradients for the coefficients.
#' @export
gradient_R <- function(X, y, beta) {
  n <- nrow(X)
  p <- ncol(X)
  grad <- numeric(p)
  for (j in 1:p) {
    grad[j] <- 0
    for (i in 1:n) {
      eta <- sum(X[i, ] * beta)
      pi <- logistic_R(eta)
      grad[j] <- grad[j] + (y[i] - pi) * X[i, j]
    }
  }
  return(-grad)
}

#' Binomial Logistic Regression using Gradient Descent
#'
#' This function fits a logistic regression model using gradient descent.
#' @param X A numeric matrix of predictors.
#' @param y A numeric vector of binary outcomes.
#' @param alpha A numeric value for the learning rate.
#' @param max_iter Maximum number of iterations for gradient descent.
#' @param tol Convergence tolerance.
#' @return A numeric vector of estimated coefficients.
#' @examples
#' set.seed(123)
#'X <- matrix(rnorm(100 * 3), ncol = 3)  # Random predictors
#'beta_true <- c(0.5, -1, 2)  # True coefficients
#'y <- rbinom(100, 1, prob = logistic_R(X %*% beta_true))  # Generate binary outcome
#'beta_est <- bino_logistic_R(X, y, alpha = 0.01, max_iter = 500, tol = 1e-6)
#'print(beta_est)

#' @export
bino_logistic_R <- function(X, y, alpha = 0.01, max_iter = 1000, tol = 1e-6) {
  p <- ncol(X)
  beta <- rep(0, p)  # Initialize coefficients
  for (iter in 1:max_iter) {
    grad <- gradient_R(X, y, beta)  # Calculate gradient
    beta_new <- beta - alpha * grad  # Update coefficients
    if (sum(abs(beta_new - beta)) < tol) {
      cat("Convergence reached at iteration", iter, "\n")
      return(beta_new)
    }
    beta <- beta_new  # Update beta
  }
  return(beta)
}


