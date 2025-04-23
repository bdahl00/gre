use rand::prelude::*;
use rand_distr::{Beta, Distribution, Normal, Poisson, StandardNormal, Uniform};
use statrs::function::*;

use crate::adaptive_params::AdaptiveParams;

pub struct GreObj {
    pub num_obs: usize,
    pub num_eigs: usize,
    pub scan_index: usize,
    pub sigma2: f64,
    pub eps_t_eps: f64,
    pub eps_norm: f64,

    pub epsilon: nalgebra::DVector<f64>,
    pub kappa: f64,
    pub s: nalgebra::DVector<f64>,
    pub lambda: nalgebra::DVector<f64>,
    pub lambda_pp: Vec<AdaptiveParams>,
    pub prior_lambda_shape: f64,
    pub prior_lambda_rate: f64,
    pub eta: nalgebra::DVector<f64>,
    pub c_eta: nalgebra::DVector<f64>,
}

impl GreObj {
    pub fn new(
        num_obs: usize,
        num_eigs: usize,
        prior_lambda_shape: f64,
        prior_lambda_rate: f64,
    ) -> Self {
        let mut lambda_pp_vec: Vec<AdaptiveParams> = Vec::with_capacity(num_eigs);
        lambda_pp_vec.resize(num_eigs, AdaptiveParams::new(0.0001)); // TODO: is the mean right?
        let mut pre_epsilon: Vec<f64> = Vec::with_capacity(num_obs);
        let mut rng = rand::thread_rng();
        let norm = Normal::new(0.0, 1.0).unwrap();
        for _index in 0..num_obs {
            pre_epsilon.push(norm.sample(&mut rng));
        }
        let pre_epsilon = nalgebra::DVector::from_vec(pre_epsilon);

        let eps_t_eps_mini = &pre_epsilon
            .rows(0, num_eigs)
            .dot(&pre_epsilon.rows(0, num_eigs));
        let eps_t_eps = eps_t_eps_mini
            + &pre_epsilon
                .rows(num_eigs, num_obs - num_eigs)
                .dot(&pre_epsilon.rows(num_eigs, num_obs - num_eigs));

        let mut pre_s = Vec::with_capacity(num_eigs);
        for index in 0..num_eigs {
            pre_s.push(pre_epsilon[index] / eps_t_eps_mini.sqrt());
        }

        let mut pre_eta = Vec::with_capacity(num_eigs);
        pre_eta.resize(num_eigs, 0.1);

        let mut pre_c_eta = pre_eta.clone();
        pre_c_eta.resize(num_obs, 0.0);

        let pre_eta = nalgebra::DVector::from_vec(pre_eta);
        let pre_c_eta = nalgebra::DVector::from_vec(pre_c_eta);

        Self {
            num_obs,
            num_eigs,
            scan_index: 1,
            sigma2: 1.0,
            eps_t_eps,
            eps_norm: eps_t_eps.sqrt(),
            epsilon: pre_epsilon,
            kappa: eps_t_eps_mini / eps_t_eps,
            s: nalgebra::DVector::from_vec(pre_s),
            lambda: nalgebra::DVector::from_vec(vec![1.0; num_eigs]),
            lambda_pp: lambda_pp_vec,
            prior_lambda_shape,
            prior_lambda_rate,
            eta: pre_eta,
            c_eta: pre_c_eta,
        }
    }

    pub fn sample_kappa(&mut self) {
        let mut rng = rand::thread_rng();
        let mut qf_term = 0.0;
        for index in 0..self.num_eigs {
            qf_term +=
                self.s[index].powi(2) * self.lambda[index] / (self.sigma2 + self.lambda[index]);
        }
        let poisson = Poisson::new(self.kappa * self.eps_t_eps * qf_term / 2.0 / self.sigma2);
        let ell = poisson.unwrap().sample(&mut rng);
        let beta = Beta::new(
            self.num_eigs as f64 / 2.0 + ell,
            (self.num_obs - self.num_eigs) as f64 / 2.0,
        );
        self.kappa = beta.unwrap().sample(&mut rng);
    }

    pub fn sample_s(&mut self) {
        let mut rng = rand::thread_rng();
        let mut zeta = Vec::with_capacity(self.num_eigs);
        for index in 0..self.num_eigs {
            zeta.push(
                self.lambda[index] / (self.sigma2 + self.lambda[index])
                    * self.kappa
                    * self.eps_t_eps
                    / 2.0
                    / self.sigma2,
            );
        }
        let zeta = nalgebra::DVector::from_vec(zeta);
        for gap in 1..(self.num_eigs) {
            for i in 0..(self.num_eigs - gap) {
                let leeway = (self.s[i].powi(2) + self.s[i + gap].powi(2)).sqrt();
                if zeta[i] == zeta[i + gap] {
                    let z = nalgebra::Vector2::new(
                        rng.sample::<f64, StandardNormal>(StandardNormal),
                        rng.sample::<f64, StandardNormal>(StandardNormal),
                    );
                    let recip = 1.0 / &z.dot(&z).sqrt() * leeway;
                    let partial_s = z * recip;
                    self.s[i] = partial_s[0];
                    self.s[i + gap] = partial_s[1];
                } else {
                    let indices = nalgebra::Vector2::new(i, i + gap);
                    let rel_lambdas = nalgebra::Vector2::new(self.lambda[i], self.lambda[i + gap]);
                    // let imin = indices.imin();
                    let imin = rel_lambdas.imin();
                    let new_val = rtruncnorm(
                        (2.0 * (zeta[i] - zeta[i + gap]).abs()).sqrt(),
                        -leeway,
                        leeway,
                    );
                    self.s[indices[imin]] = new_val;
                    self.s[indices[1 - imin]] = (leeway.powi(2) - new_val.powi(2)).sqrt();
                }
            }
        }
    }

    pub fn update_lambda_i(&mut self, new_lambda_i: f64, index: usize) {
        self.lambda[index] = new_lambda_i;
        self.lambda_pp[index].update(new_lambda_i, self.scan_index);
    }

    pub fn sample_lambda(&mut self) {
        for index in 0..self.num_eigs {
            let prop_lambda_i = self.lambda_pp[index].sample();
            if prop_lambda_i < 0.0 {
                self.update_lambda_i(self.lambda[index], index);
                continue;
            }

            let curr_lambda_i = self.lambda[index];
            let curr_l_post = -(self.sigma2 + curr_lambda_i).ln() / 2.0
                + self.kappa * self.eps_t_eps * curr_lambda_i * self.s[index].powi(2)
                    / (self.sigma2 + curr_lambda_i)
                    / 2.0
                    / self.sigma2
                + (self.prior_lambda_shape - 1.0) * curr_lambda_i.ln() // This is the contribution from the prior
                - self.prior_lambda_rate * curr_lambda_i;
            let prop_l_post = -(self.sigma2 + prop_lambda_i).ln() / 2.0
                + self.kappa * self.eps_t_eps * prop_lambda_i * self.s[index].powi(2)
                    / (self.sigma2 + curr_lambda_i)
                    / 2.0
                    / self.sigma2
                + (self.prior_lambda_shape - 1.0) * curr_lambda_i.ln() // Again, from the prior
                - self.prior_lambda_rate * curr_lambda_i;
            let trans_kern_diff = (prop_lambda_i + curr_lambda_i
                - 2.0 * self.lambda_pp[index].mean)
                * (prop_lambda_i - curr_lambda_i)
                / 2.0
                / self.lambda_pp[index].variance;
            let mut rng = rand::thread_rng();
            if prop_l_post - curr_l_post + trans_kern_diff
                > Uniform::new(0.0_f64, 1.0_f64).sample(&mut rng).ln()
            {
                self.update_lambda_i(prop_lambda_i, index);
            } else {
                self.update_lambda_i(curr_lambda_i, index);
            }
        }
    }

    pub fn sample_eta(&mut self) {
        let mut rng = rand::thread_rng();
        for index in 0..self.num_eigs {
            // The name variance is a little misleading - needs to be scaled by self.sigma2
            let variance = self.lambda[index] / (self.sigma2 + self.lambda[index]);
            let mean = self.kappa.sqrt() * self.eps_norm * variance * self.s[index];
            let norm = Normal::new(mean, self.sigma2.sqrt() * variance.sqrt()).unwrap();
            self.eta[index] = norm.sample(&mut rng);
        }
    }

    pub fn sample_c_eta(&mut self) {
        let eta_norm = *&self.eta.dot(&self.eta).sqrt();
        let cos_phi = self.kappa.sqrt() * &self.s.dot(&self.eta) / eta_norm;
        let center = &self.epsilon * cos_phi * eta_norm / self.eps_norm; // Don't know why we have to dereference
        let mut rng = rand::thread_rng();
        let mut z: Vec<f64> = Vec::with_capacity(self.num_obs);
        for _index in 0..self.num_obs {
            z.push(rng.sample(StandardNormal));
        }
        let z = nalgebra::DVector::from_vec(z);
        let orth_proj = &z - &self.epsilon * *&self.epsilon.dot(&z) / self.eps_t_eps; // O(n)
        let orth_proj = &orth_proj / *&orth_proj.dot(&orth_proj).sqrt(); // O(n)
        self.c_eta = &center + &orth_proj * eta_norm * (1.0 - cos_phi.powi(2)).sqrt();
    }

    pub fn update_epsilon_kappa_and_s(&mut self, new_epsilon: Vec<f64>) {
        let new_epsilon: nalgebra::DVector<f64> = nalgebra::DVector::from_vec(new_epsilon);
        let ne_t_ne = *&new_epsilon.dot(&new_epsilon); // O(n)
        let ne_norm = ne_t_ne.sqrt();
        let s_t_eta = *&self.eta.dot(&self.s);
        let eps_t_ne = *&self.epsilon.dot(&new_epsilon); // O(n)

        // Mutable useful quantities
        let mut a: f64 = 1.0;
        let mut b = self.kappa.sqrt() * s_t_eta;
        let mut d = *&self.eta.dot(&self.eta);
        let mut x = eps_t_ne / self.eps_norm;
        let mut y = *&self.c_eta.dot(&new_epsilon); // O(n)
        let mut omega_t_omega: f64 = 0.0;

        let mut pre_from_b = self.kappa.sqrt() * self.s[0];
        let mut from_a = self.kappa * self.s[0].powi(2); // Is it better to square pre_from_b?
        let mut from_b = pre_from_b * self.eta[0];
        let mut from_d = self.eta[0].powi(2);

        let mut omega: Vec<f64> = Vec::with_capacity(self.num_eigs);
        //omega.resize(self.num_eigs, 0.0);
        let mut rng = rand::thread_rng();

        // Sampling omega[0]
        let mut denom = a * d - b.powi(2);

        let mut tau_t_tau = (d * from_a - 2.0 * b * from_b + a * from_d) / denom;
        let mut tau_t_ne = ((d * pre_from_b - b * self.eta[0]) * x
            + (a * self.eta[0] - b * pre_from_b) * y)
            / denom;
        let mut quad_form = (x.powi(2) * d - 2.0 * x * y * b + y.powi(2) * a) / denom;

        let beta = Beta::new(0.5, (self.num_obs - 3) as f64 / 2.0);
        let mut cos_phi = beta.unwrap().sample(&mut rng).sqrt();
        omega.push(tau_t_ne + (1.0 - tau_t_tau).sqrt() * (ne_t_ne - quad_form).sqrt() * cos_phi);

        // Sampling omega[i]
        for index in 1..(self.num_eigs - 1) {
            a -= from_a;
            b -= from_b;
            d -= from_d;
            x -= self.kappa.sqrt() * self.s[index - 1] * omega[index - 1];
            y -= self.eta[index - 1] * omega[index - 1];
            omega_t_omega += omega[index - 1].powi(2);

            pre_from_b = self.kappa.sqrt() * self.s[index];
            from_a = self.kappa * self.s[index].powi(2);
            from_b = pre_from_b * self.eta[index];
            from_d = self.eta[index].powi(2);

            denom = a * d - b.powi(2);
            tau_t_tau = (d * from_a - 2.0 * b * from_b + a * from_d) / denom;
            tau_t_ne = ((d * pre_from_b - b * self.eta[index]) * x
                + (a * self.eta[index] - b * pre_from_b) * y)
                / denom;
            quad_form = (x.powi(2) * d - 2.0 * x * y * b + y.powi(2) * a) / denom;

            let beta = Beta::new(0.5, (self.num_obs - 3 - index) as f64 / 2.0);
            cos_phi = beta.unwrap().sample(&mut rng).sqrt();
            omega.push(
                tau_t_ne
                    + (1.0 - tau_t_tau).sqrt()
                        * (ne_t_ne - omega_t_omega - quad_form).sqrt()
                        * cos_phi,
            );
        }

        a -= from_a;
        b -= from_b;
        d -= from_d;
        x -= self.kappa.sqrt() * self.s[self.num_eigs - 2] * omega[self.num_eigs - 2];
        y -= self.eta[self.num_eigs - 2] * omega[self.num_eigs - 2];
        omega_t_omega += omega[self.num_eigs - 2].powi(2);

        denom = a * d - b.powi(2);
        omega.push(
            ((d * self.kappa.sqrt() * self.s[self.num_eigs - 1] - b * self.eta[self.num_eigs - 1])
                * x
                + (a * self.eta[self.num_eigs - 1]
                    - b * self.kappa.sqrt() * self.s[self.num_eigs - 1])
                    * y)
                / denom,
        );
        omega_t_omega += omega[self.num_eigs - 1].powi(2);

        let omega = nalgebra::DVector::from_vec(omega);

        self.kappa = omega_t_omega / ne_t_ne;
        self.s = omega / omega_t_omega.sqrt();
        self.epsilon = new_epsilon;
        self.eps_t_eps = ne_t_ne;
        self.eps_norm = ne_norm;
    }

    // Interface functions
    pub fn run_sampler(&mut self) {
        self.sample_kappa();
        self.sample_s();
        self.sample_lambda();
        self.sample_eta();
        self.sample_c_eta();
        self.scan_index += 1;
    }

    pub fn update_sigma2(&mut self, resid_var: f64) {
        self.sigma2 = resid_var;
    }

    pub fn get_c_eta(&self) -> &nalgebra::DVector<f64> {
        &self.c_eta
    }

    // Testing only - to be deleted - should never be used except for validation
    pub fn get_kappa(&self) -> f64 {
        self.kappa
    }

    pub fn get_s(&self) -> &nalgebra::DVector<f64> {
        &self.s
    }

    pub fn get_lambda(&self) -> &nalgebra::DVector<f64> {
        &self.lambda
    }

    pub fn get_eta(&self) -> &nalgebra::DVector<f64> {
        &self.eta
    }

    // This function should only be called on R output
    // Otherwise things won't match up (norms of eta and c_eta, for example)
    // and we'll get errors, and that won't help anything
    pub fn strongarm_update_everything(
        &mut self,
        new_kappa: f64,
        new_s: Vec<f64>,
        new_lambda: Vec<f64>,
        new_eta: Vec<f64>,
        new_c_eta: Vec<f64>,
    ) {
        self.kappa = new_kappa;
        self.s = nalgebra::DVector::from_vec(new_s);
        self.lambda = nalgebra::DVector::from_vec(new_lambda);
        self.eta = nalgebra::DVector::from_vec(new_eta);
        self.c_eta = nalgebra::DVector::from_vec(new_c_eta);
    }
}

fn pnorm(x: f64) -> f64 {
    erf::erfc(-x / 2.0_f64.sqrt()) / 2.0
}

fn qnorm(p: f64) -> f64 {
    -2.0_f64.sqrt() * erf::erfc_inv(2.0 * p)
}

fn rtruncnorm(sd: f64, a: f64, b: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(pnorm(a / sd), pnorm(b / sd));
    let u = uniform.sample(&mut rng);
    qnorm(u) * sd
}
