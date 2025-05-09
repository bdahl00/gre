% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/update_sigma2.R
\name{update_sigma2}
\alias{update_sigma2}
\title{Update Residual Variance}
\usage{
update_sigma2(ptr, new_sigma2)
}
\arguments{
\item{ptr}{An external pointer created by the initialize_gre() function.}

\item{new_sigma2}{The new residual variance}
}
\value{
Nothing, ideally
}
\description{
This function updates the residual variance of a gre object. Note that
no function in the gre library calculates residual variances; those must
instead be calculated externally, just as the estimate of the main effect is.
}
\examples{
update_sigma2(greObj, 2)
}
