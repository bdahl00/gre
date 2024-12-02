#' Update Residual Variance
#' 
#' This function updates the residual variance of a gre object. Note that
#' no function in the gre library calculates residual variances; those must
#' instead be calculated externally, just as the estimate of the main effect is.
#'
#' @param ptr An external pointer created by the initialize_gre() function.
#' @param new_sigma2 The new residual variance
#' 
#' @return Nothing, ideally
#'
#' @export
#' @examples
#' update_sigma2(greObj, 2)
update_sigma2 <- function(ptr, new_sigma2) {
  .Call(.update_sigma2, ptr, new_sigma2)
}
