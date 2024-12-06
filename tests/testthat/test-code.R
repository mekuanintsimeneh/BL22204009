

test_that("gibbs_exponential_cpp works as expected", {
  set.seed(123)
  y <- rexp(100, rate = 2)  # Generate data with true lambda = 2
  samples <- gibbs_exponential_cpp(y, alpha = 1, beta = 1, n_iter = 1000)

  expect_length(samples, 1000)
  expect_true(all(samples > 0))  # Lambda must be positive
  expect_true(abs(mean(samples) - 2) < 0.5)  # Check approximate mean
})


test_that("Bayesian Poisson-Gamma Gibbs sampling returns valid posterior samples", {
  set.seed(123)
  data <- c(2, 4, 1, 3, 5, 2, 4, 3)
  alpha_prior <- 2
  beta_prior <- 1
  n_iter <- 1000

  # Run the function
  samples <- bayesian_poisson_gamma_R(data, alpha_prior, beta_prior, n_iter)

  # Check that the output is a numeric vector
  expect_type(samples, "double")

  # Check that the number of samples matches n_iter
  expect_equal(length(samples), n_iter)

  # Ensure posterior samples are positive
  expect_true(all(samples > 0))
})
test_that("Logistic function maps input to valid probability range", {
  x <- c(-100, -1, 0, 1, 100)

  # Run the logistic function
  result <- logistic_R(x)

  # Check that all values are between 0 and 1
  expect_true(all(result >= 0 & result <= 1))

  # Check specific cases
  expect_equal(logistic_R(0), 0.5)
  expect_equal(logistic_R(-Inf), 0)
  expect_equal(logistic_R(Inf), 1)
})
test_that("Log-likelihood for binomial logistic regression is computed correctly", {
  set.seed(123)
  X <- matrix(rnorm(10 * 2), ncol = 2)
  beta <- c(0.5, -1)
  y <- rbinom(10, 1, prob = logistic_R(X %*% beta))

  # Compute the log-likelihood
  ll <- log_likelihood_R(X, y, beta)

  # Check that the log-likelihood is a single numeric value
  expect_length(ll, 1)
  expect_type(ll, "double")
})
test_that("Gradient of the log-likelihood is computed correctly", {
  set.seed(123)
  X <- matrix(rnorm(10 * 2), ncol = 2)
  beta <- c(0.5, -1)
  y <- rbinom(10, 1, prob = logistic_R(X %*% beta))

  # Compute the gradient
  grad <- gradient_R(X, y, beta)

  # Check that the gradient is a numeric vector with the correct length
  expect_type(grad, "double")
  expect_equal(length(grad), ncol(X))
})
test_that("Binomial logistic regression using gradient descent converges", {
  set.seed(123)
  X <- matrix(rnorm(100 * 3), ncol = 3)
  beta_true <- c(0.5, -1, 2)
  y <- rbinom(100, 1, prob = logistic_R(X %*% beta_true))

  # Fit the model
  beta_est <- bino_logistic_R(X, y, alpha = 0.01, max_iter = 500, tol = 1e-6)

  # Check that the output is a numeric vector with the correct length
  expect_type(beta_est, "double")
  expect_equal(length(beta_est), ncol(X))

  # Check that the estimated coefficients are reasonably close to the true values
  expect_true(mean(abs(beta_est - beta_true)) < 0.5)
})
