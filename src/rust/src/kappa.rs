use rand_distr::{Distribution, Uniform};
use crate::adaptive_params::AdaptiveParams;
use crate::gre::GreObj;

pub struct Kappa {
    pub value: f64,
    prop_params: AdaptiveParams,
}

impl Kappa {
    pub fn new(min_var: f64) -> Self {
        Self{value: 0.5, prop_params: AdaptiveParams::new(min_var)}
    }
    
    pub fn sample(&mut self, gre_obj: GreObj) -> GreObj {
        let prop_kappa = self.prop_params.sample();
        if prop_kappa < 0.0 || 1.0 < prop_kappa {self.update(self.value, gre_obj.scan_index);}
        
        // Calculate difference in log posteriors
        let mut zeta = Vec::with_capacity(gre_obj.num_eigs);
        for index in 0..gre_obj.num_eigs {
            zeta[index] = gre_obj.lambda.value[index] / (gre_obj.sigma2 + gre_obj.lambda.value[index]);
        }
        let zeta = nalgebra::DMatrix::from_vec(gre_obj.num_eigs, 1, zeta);
        let llik_prod = gre_obj.sum_of_squared_epsilons + (zeta.transpose() * gre_obj.s.value.clone())[(0, 0)] / 2.0 / gre_obj.sigma2;
        let llik_diff = (prop_kappa - self.value) * llik_prod;
        let prior_diff = (gre_obj.num_eigs as f64 / 2.0 - 1.0) * (prop_kappa.ln() - self.value.ln()) +
            ((gre_obj.num_obs - gre_obj.num_eigs) as f64 / 2.0 - 1.0) *
                (((1.0 - prop_kappa) as f64).ln() - (1.0 - self.value).ln());
        let trans_kern_diff = (prop_kappa + self.value - 2.0 * self.prop_params.mean) * (prop_kappa - self.value) / 2.0 / self.prop_params.variance;
        let mut rng = rand::thread_rng();
        if llik_diff + prior_diff + trans_kern_diff > Uniform::new(0 as f64, 1 as f64).sample(&mut rng) {self.update(prop_kappa, gre_obj.scan_index);}
        else {self.update(self.value, gre_obj.scan_index);}
        gre_obj
    }

    pub fn update(&mut self, kappa: f64, scan_index: usize) {
        self.value = kappa;
        self.prop_params.update(kappa, scan_index);
    }
}

impl GreObj {
    pub fn sample_kappa(mut self) -> Self {

        let prop_kappa = self.kappa.prop_params.sample();
        if prop_kappa < 0.0 || 1.0 < prop_kappa {
            self.kappa.prop_params.update(self.kappa.value, &self.scan_index);
        }
        
        // Calculate difference in log posteriors
        let mut zeta = Vec::with_capacity(self.num_eigs);
        for index in 0..self.num_eigs {
            zeta[index] = self.lambda.value[index] / (self.sigma2 + self.lambda.value[index]);
        }
        let zeta = nalgebra::DMatrix::from_vec(self.num_eigs, 1, zeta);
        let llik_prod = self.sum_of_squared_epsilons + (zeta.transpose() * self.s.value.clone())[(0, 0)] / 2.0 / self.sigma2;
        let llik_diff = (prop_kappa - self.kappa.value) * llik_prod;
        let prior_diff = (self.num_eigs as f64 / 2.0 - 1.0) * (prop_kappa.ln() - self.kappa.value.ln()) +
            ((self.num_obs - self.num_eigs) as f64 / 2.0 - 1.0) *
                (((1.0 - prop_kappa) as f64).ln() - (1.0 - self.kappa.value).ln());
        let trans_kern_diff = (prop_kappa + self.kappa.value - 2.0 * self.kappa.prop_params.mean) * (prop_kappa - self.kappa.value) / 2.0 / self.kappa.prop_params.variance;
        let mut rng = rand::thread_rng();
        if llik_diff + prior_diff + trans_kern_diff > Uniform::new(0 as f64, 1 as f64).sample(&mut rng) {self.update_kappa(prop_kappa, self.scan_index);}
        else {self.update_kappa(self.kappa.value, self.scan_index);}
        self
    }

    pub fn update_kappa(mut self, kappa: f64, scan_index: &usize) -> Self {
        self.kappa.value = kappa;
        self.kappa.prop_params.update(kappa, scan_index);
        self
    }
}
