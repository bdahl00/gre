use rand_distr::{Distribution, Uniform, Gamma};
use crate::adaptive_params::AdaptiveParams;
use crate::gre::GreObj;

pub struct S {
    pub value: nalgebra::DMatrix<f64>,
    pub mean_vec: nalgebra::DMatrix<f64>,
    pub alpha: nalgebra::DMatrix<f64>,
    pub prop_params: AdaptiveParams,
}

impl S {
    pub fn new(num_eigs: usize, min_var: f64) -> Self {
        Self{value: nalgebra::DMatrix::from_vec(num_eigs, 1, vec![1.0 / (num_eigs as f64); num_eigs]),
             mean_vec: nalgebra::DMatrix::from_vec(num_eigs, 1, vec![1.0 / (num_eigs as f64); num_eigs]),
             alpha: nalgebra::DMatrix::from_vec(num_eigs, 1, vec![1.0; num_eigs]),
             prop_params: AdaptiveParams::new(min_var)}
    }    

    pub fn sample(&mut self, gre_obj: &GreObj) {
        let prop_s = self.draw_dirichlet(gre_obj);
        let mut zeta = Vec::with_capacity(gre_obj.num_eigs);
        let mut trans_kern_diff: f64 = 0.0;
        for index in 0..gre_obj.num_eigs {
            zeta[index] = gre_obj.lambda.value[index] / (gre_obj.sigma2 + gre_obj.lambda.value[index]);
            trans_kern_diff += (self.alpha[(index, 0)] - 1.0) * (self.value[(index, 0)].ln() - prop_s[(index, 0)].ln());
        }
        let zeta = nalgebra::DMatrix::from_vec(gre_obj.num_eigs, 1, zeta);
        let llik_diff = gre_obj.kappa.value * gre_obj.sum_of_squared_epsilons / 2.0 / gre_obj.sigma2 * (zeta.transpose() * (prop_s.clone() - self.value.clone()))[(0, 0)];
        let prior_diff = gre_obj.num_eigs as f64 * 
            (((self.value.transpose() * self.value.clone())[(0, 0)]).ln() - ((prop_s.clone().transpose() * prop_s.clone())[(0, 0)]).ln());
        let mut rng = rand::thread_rng();
        if llik_diff + prior_diff + trans_kern_diff > Uniform::new(0 as f64, 1 as f64).sample(&mut rng) {self.update(prop_s.clone(), gre_obj.scan_index);}
        else {self.update(self.value.clone(), gre_obj.scan_index);}
    }

    pub fn draw_dirichlet(&self, gre_obj: &GreObj) -> nalgebra::DMatrix<f64> {
        let mut dir_vec = nalgebra::DMatrix::from_vec(gre_obj.num_eigs, 1, vec![0.0; gre_obj.num_eigs]);
        let mut rng = rand::thread_rng();
        let mut element_sum: f64 = 0.0;
        for index in 0..gre_obj.num_eigs {
            let gamma = Gamma::new(self.alpha[(index, 0)], 1.0);
            dir_vec[(index, 1)] = gamma.unwrap().sample(&mut rng);
            element_sum += dir_vec[(index, 0)];
        }
        dir_vec / element_sum
    }

    pub fn update(&mut self, s: nalgebra::DMatrix<f64>, scan_index: usize) {
        self.value = s;
        self.mean_vec = ((scan_index - 1) as f64 * self.mean_vec.clone() + self.value.clone()) / scan_index as f64; // Ask Dad - is this right?
        self.prop_params.update(self.value[0], scan_index);
        self.alpha = (self.prop_params.mean * (1.0 - self.prop_params.mean) / self.prop_params.variance - 1 as f64) * self.mean_vec.clone();
    }
}
