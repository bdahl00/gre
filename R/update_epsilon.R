#' Update (Correlated) Residual
#'
#' This function internally updates a gre external pointer's residual and
#' adjusts the kappa and s parameters as described in (cite). Nothing is
#' returned, but the state is stored internally. The length of new_epsilon
#' must agree with the number of observations the sampler was initialized
#' to have.
#'
#' @param ptr An external pointer created by the initialize_gre() function.
#' @param new_epsilon A numeric vector of (correlated) residuals.
#'
#' @return Nothing.
#' @export
#' @examples
#' update_epsilon(greObj, rnorm(100)) # If the sampler has 100 observations
update_epsilon <- function(ptr, new_epsilon) {
  invisible(.Call(.update_epsilon, ptr, new_epsilon))
}
