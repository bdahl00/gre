#' Extract Correlation Adjustment
#'
#' This function extracts the correlation adjustment C times eta in
#' (cite here too, I guess). When implementing a sampler according to
#' (cite here too), this vector should be subtracted from the response
#' before the mean function contribution Gibbs step is entered.
#'
#' @param ptr An external pointer created by the initialize_gre() function.
#'
#' @return A numeric vector.
#' @export
#' @examples
#' get_c_eta(greObj)
get_c_eta <- function(ptr) {
  .Call(.get_c_eta, ptr)
}
