use rand_distr::{Distribution, Uniform};
use crate::adaptive_params::AdaptiveParams;
use crate::gre::GreObj;

struct LambdaI {
    value: f64,
    eig_index: usize,
    prop_params: AdaptiveParams,
}

impl LambdaI {
    pub fn new(eig_index: usize, min_var: f64) -> Self {
        Self{value: 1.0,
             eig_index,
             prop_params: AdaptiveParams::new(min_var),}
    }

    pub fn sample(&mut self, gre_obj: &GreObj) {
        let prop_lambda_i = self.prop_params.sample();
        if prop_lambda_i < 0.0 {
            self.update(self.value, gre_obj.scan_index);
            return;
        }        

        let curr_l_post = -(gre_obj.sigma2 + self.value).ln() / 2.0 +
            gre_obj.kappa.value * gre_obj.sum_of_squared_epsilons * self.value * gre_obj.s.value[(self.eig_index, 0)] / (gre_obj.sigma2 + self.value) / 2.0 / gre_obj.sigma2;
        let prop_l_post = -(gre_obj.sigma2 + prop_lambda_i).ln() / 2.0 + 
            gre_obj.kappa.value * gre_obj.sum_of_squared_epsilons * prop_lambda_i * gre_obj.s.value[(self.eig_index, 0)] / (gre_obj.sigma2 + prop_lambda_i) / 2.0 / gre_obj.sigma2;
        let prior_diff = (gre_obj.lambda.prior_shape - 1.0) * (prop_lambda_i.ln() - self.value.ln()) - gre_obj.lambda.prior_rate * (prop_lambda_i - self.value);
        let trans_kern_diff = (prop_lambda_i + self.value - 2.0 * self.prop_params.mean) * (prop_lambda_i - self.value) / 2.0 / self.prop_params.variance;
        let mut rng = rand::thread_rng();
        if prop_l_post - curr_l_post + prior_diff + trans_kern_diff > Uniform::new(0 as f64, 1 as f64).sample(&mut rng) {self.update(prop_lambda_i, gre_obj.scan_index);}
        else {self.update(self.value, gre_obj.scan_index);}
    }

    pub fn update(&mut self, lambda_i: f64, scan_index: usize) {
        self.value = lambda_i;
        self.prop_params.update(lambda_i, scan_index);
    }
}

pub struct Lambda {
    pub value: nalgebra::DMatrix<f64>,
    lambda_vec: Vec<LambdaI>,
    prior_shape: f64,
    prior_rate: f64,
}

impl Lambda {
    pub fn new(num_eigs: usize, prior_shape: f64, prior_rate: f64, min_var: f64) -> Self {
        let mut lambda_i_vec = Vec::with_capacity(num_eigs); // Ask Dad
        for index in 0..num_eigs {
            lambda_i_vec[index] = LambdaI::new(index, min_var); // Ask Dad how to initialize something
        }
        Self{value: nalgebra::DMatrix::from_vec(num_eigs, 1, vec![1.0; num_eigs]),
             lambda_vec: lambda_i_vec,
             prior_shape,
             prior_rate}
    }

    pub fn sample(&mut self, gre_obj: &GreObj) {
        for index in 0..gre_obj.num_obs {
            self.lambda_vec[index].sample(gre_obj);
            self.value[(index, 0)] = self.lambda_vec[index].value;
        }
    }
}
