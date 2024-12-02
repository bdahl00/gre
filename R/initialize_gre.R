#' Initialize a gre External Pointer
#'
#' This function initializes an object implementing the method of
#' (cite the paper here, I guess). The state of the object is stored
#' internally.
#'
#' @param n An integer, the number of observations.
#' @param m An integer, the number of eigenvalues to consider.
#' @param prior_lambda_shape A scalar, by default 2.
#' @param prior_lambda_rate A scalar, by default 2.
#'
#' @return An external pointer.
#' @export
#' @examples
#' initialize_gre(1000, 100, 2, 2)
initialize_gre <- function(n, m, prior_lambda_shape, prior_lambda_rate) {
    .Call(.initialize_gre, n, m, prior_lambda_shape, prior_lambda_rate)
}
