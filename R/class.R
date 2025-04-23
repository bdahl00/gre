methods::setClass(
  "gre",
  slots = list(
    pointer = "externalptr"
  )
)

# new()

gre <- setRefClass(
  "gre",
  fields = list(
    pointer = "externalptr"
  ),
  methods = list(
    initialize = 
      function(n, m, prior_lambda_shape, prior_lambda_rate) {
        .Call(.initialize_gre, n, m, prior_lambda_shape, prior_lambda_rate)
      },
    update_sigma2 =
      function(new_sigma2) {
        invisible(.Call(.update_sigma2, .self$pointer, new_sigma2))
      },
    update_epsilon = 
      function(new_epsilon) {
        invisible(.Call(.update_epsilon, .self$pointer, new_epsilon))
      },
    get_c_eta = 
      function() {
        .Call(.get_c_eta, .self$pointer)
      },
    run = 
      function() {
        invisible(.Call(.run, .self$pointer))
      }
    )
  )
