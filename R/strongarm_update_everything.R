#' Extract Correlation Adjustment
#'
#' This function extracts the correlation adjustment C times eta in
#' (cite here too, I guess). When implementing a sampler according to
#' (cite here too), this vector should be subtracted from the response
#' before the mean function contribution Gibbs step is entered.
#'
#' @param ptr An external pointer created by the initialize_gre() function.
#'
#' @return A scalar.
#' @export
#' @examples
#' get_c_eta(greObj)
strongarm_update_everything <- function(ptr, kappa, s, lambda, eta, c_eta) {
  invisible(.Call(.strongarm_update_everything, ptr, kappa, s, lambda, eta, c_eta))
}
