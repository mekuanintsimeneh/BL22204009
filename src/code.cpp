#include <Rcpp.h>
using namespace Rcpp;

//' Gibbs Sampling for Exponential Distribution (Rcpp)
 //'
 //' Performs Gibbs sampling for the rate parameter (lambda) of an exponential distribution.
 //'
 //' @param y Numeric vector of observed data.
 //' @param alpha Prior shape parameter for the Gamma distribution.
 //' @param beta Prior rate parameter for the Gamma distribution.
 //' @param n_iter Number of Gibbs sampling iterations.
 //' @return A NumericVector of sampled lambda values.
 //' @examples
 //' set.seed(123)
 //' y <- rexp(100, rate = 2)
 //' samples <- gibbs_exponential_cpp(y, alpha = 1, beta = 1, n_iter = 1000)
 //' hist(samples, breaks = 30, main = "Posterior Samples of Lambda")
 //' @export
 // [[Rcpp::export]]
 NumericVector gibbs_exponential_cpp(NumericVector y, double alpha, double beta, int n_iter) {
   int n = y.size();                  // Number of observations
   double sum_y = sum(y);             // Sufficient statistic
   NumericVector lambda_samples(n_iter); // Output vector for lambda samples

   for (int i = 0; i < n_iter; ++i) {
     // Update posterior parameters
     double alpha_post = alpha + n;
     double beta_post = beta + sum_y;

     // Sample from Gamma distribution
     lambda_samples[i] = R::rgamma(alpha_post, 1.0 / beta_post); // Gamma(shape, scale)
   }

   return lambda_samples;
 }


 //' Gibbs Sampling for Poisson-Gamma Model (Rcpp)
 //'
 //' Performs Gibbs sampling for the rate parameter (lambda) of a Poisson likelihood with a Gamma prior.
 //' The model assumes that the data follow a Poisson distribution, and the prior on lambda is a Gamma distribution.
 //'
 //' @param data Numeric vector of observed counts (Poisson data).
 //' @param alpha_prior Prior shape parameter for the Gamma distribution.
 //' @param beta_prior Prior rate parameter for the Gamma distribution.
 //' @param n_iter Number of Gibbs sampling iterations.
 //' @return A NumericVector of sampled lambda values.
 //' @examples
 //' set.seed(123)
 //' data <- c(2, 4, 1, 3, 5, 2, 4, 3)
 //' samples <- bayesian_poisson_gamma_cpp(data, alpha_prior = 2, beta_prior = 1, n_iter = 10000)
 //' hist(samples, breaks = 30, main = "Posterior Samples of Lambda (Poisson Rate)")
 //' @export
 // [[Rcpp::export]]
 NumericVector bayesian_poisson_gamma_cpp(NumericVector data, double alpha_prior, double beta_prior, int n_iter = 10000) {
   int n = data.size();               // Number of observations
   double sum_data = sum(data);       // Sum of observed data

   NumericVector lambda_samples(n_iter);  // Vector to store posterior samples

   // Initial guess for lambda
   double lambda_current = 1.0;

   // Gibbs sampling loop
   for (int i = 0; i < n_iter; i++) {
     // Posterior shape parameter is alpha + sum(y)
     double posterior_shape = alpha_prior + sum_data;

     // Posterior scale parameter is beta_prior + n (number of data points)
     double posterior_scale = 1.0 / (1.0 / beta_prior + n);

     // Sample lambda from the posterior Gamma distribution
     lambda_current = R::rgamma(posterior_shape, posterior_scale);

     // Store the sampled value of lambda
     lambda_samples[i] = lambda_current;
   }

   return lambda_samples;
 }

 //' Logistic function in C++
 //'
 //' This is a logistic sigmoid function that maps input values to probabilities between 0 and 1.
 //' @param x A numeric value or vector.
 //' @return Logistic transformation of input x.
 //' @export
 // [[Rcpp::export]]
 NumericVector logistic_cpp(NumericVector x) {
   return 1 / (1 + exp(-x));
 }


 //' Log-likelihood for Logistic Regression in C++
 //'
 //' This function computes the log-likelihood for logistic regression given a set of data and coefficients.
 //' @param X A numeric matrix of predictors.
 //' @param y A numeric vector of binary outcomes (0 or 1).
 //' @param beta A numeric vector of regression coefficients.
 //' @return The log-likelihood value.
 //' @export
 // [[Rcpp::export]]
 double log_likelihood_cpp(NumericMatrix X, NumericVector y, NumericVector beta) {
   int n = X.nrow();
   int p = X.ncol();
   double ll = 0.0;

   for (int i = 0; i < n; ++i) {
     double eta = 0.0;
     for (int j = 0; j < p; ++j) {
       eta += X(i, j) * beta[j];
     }
     double pi = 1.0 / (1.0 + exp(-eta)); // Logistic function
     ll += y[i] * log(pi) + (1 - y[i]) * log(1 - pi);
   }
   return -ll;
 }


 //' Gradient of the log-likelihood for Binomial Logistic Regression in C++
 //'
 //' This function computes the gradient of the log-likelihood for logistic regression.
 //' @param X A numeric matrix of predictors.
 //' @param y A numeric vector of binary outcomes (0 or 1).
 //' @param beta A numeric vector of regression coefficients.
 //' @return A numeric vector of gradients for the coefficients.
 //' @export
 // [[Rcpp::export]]
 NumericVector gradient_cpp(NumericMatrix X, NumericVector y, NumericVector beta) {
   int n = X.nrow();
   int p = X.ncol();
   NumericVector grad(p);

   for (int j = 0; j < p; ++j) {
     grad[j] = 0;
     for (int i = 0; i < n; ++i) {
       double eta = 0;
       for (int k = 0; k < p; ++k) {
         eta += X(i, k) * beta[k];
       }
       double pi = logistic_cpp(eta)[0];
       grad[j] += (y[i] - pi) * X(i, j);
     }
   }
   return -grad;
 }


 //' Binomial Logistic Regression using Gradient Descent in C++
 //'
 //' This function fits a logistic regression model using gradient descent.
 //' @param X A numeric matrix of predictors.
 //' @param y A numeric vector of binary outcomes.
 //' @param alpha A numeric value for the learning rate.
 //' @param max_iter Maximum number of iterations for gradient descent.
 //' @param tol Convergence tolerance.
 //' @return A numeric vector of estimated coefficients.
 //' @export
 // [[Rcpp::export]]
 NumericVector bino_logistic_cpp(NumericMatrix X, NumericVector y, double alpha = 0.01, int max_iter = 1000, double tol = 1e-6) {
   int p = X.ncol();
   NumericVector beta(p);
   beta = 0;  // Initialize coefficients to zero

   for (int iter = 0; iter < max_iter; ++iter) {
     NumericVector grad = gradient_cpp(X, y, beta);  // Calculate gradient
     NumericVector beta_new = beta - alpha * grad;  // Update coefficients

     // Check for convergence
     if (sum(abs(beta_new - beta)) < tol) {
       Rcout << "Convergence reached at iteration " << iter << "\n";
       return beta_new;
     }
     beta = beta_new;  // Update beta
   }
   return beta;
 }



