use rand::prelude::*;
use rand_distr::num_traits;
use rand_distr::num_traits::Pow;
use rand_distr::{Beta, Distribution, Gamma, Normal, StandardNormal, Uniform};
//use crate::kappa::Kappa;
//use crate::s::S;
//use crate::lambda::Lambda;
use crate::adaptive_params::AdaptiveParams;

pub struct GreObjOld {
    pub num_obs: usize,
    pub num_eigs: usize,
    pub scan_index: usize,
    pub sigma2: f64,
    pub eps_t_eps: f64,
    pub eps_norm: f64,

    pub epsilon: nalgebra::DMatrix<f64>,
    //pub kappa: Kappa,
    pub kappa: f64,
    pub kappa_pp: AdaptiveParams,
    //pub s: S,
    pub s: nalgebra::DMatrix<f64>,
    pub s_mean_vec: nalgebra::DMatrix<f64>,
    pub s_alpha: nalgebra::DMatrix<f64>,
    pub s_pp: AdaptiveParams,

    pub sqrt_s: nalgebra::DMatrix<f64>,
    //pub lambda: Lambda, // I have to be very careful handling this, since I'm collapsing two structs
    pub lambda: nalgebra::DMatrix<f64>,
    pub lambda_pp: Vec<AdaptiveParams>,
    pub prior_lambda_shape: f64,
    pub prior_lambda_rate: f64,
    pub eta: nalgebra::DMatrix<f64>,
    pub c_eta: nalgebra::DMatrix<f64>,
}

impl GreObjOld {
    pub fn new(
        num_obs: usize,
        num_eigs: usize,
        prior_lambda_shape: f64,
        prior_lambda_rate: f64,
    ) -> Self {
        let mut lambda_pp_vec: Vec<AdaptiveParams> = Vec::with_capacity(num_eigs);
        lambda_pp_vec.resize(num_eigs, AdaptiveParams::new(0.0001));

        let mut pre_epsilon: Vec<f64> = Vec::with_capacity(num_obs);
        let mut rng = rand::thread_rng();
        let norm = Normal::new(0.0, 1.0).unwrap();
        for _index in 0..num_obs {
            pre_epsilon.push(norm.sample(&mut rng));
        }
        //eprintln!("pre_epsilon matrix conversion reached");
        let pre_epsilon = nalgebra::DMatrix::from_vec(num_obs, 1, pre_epsilon);

        let eps_t_eps_mini = &pre_epsilon
            .rows(0, num_eigs)
            .dot(&pre_epsilon.rows(0, num_eigs));
        let eps_t_eps = eps_t_eps_mini
            + &pre_epsilon
                .rows(num_eigs, num_obs - num_eigs)
                .dot(&pre_epsilon.rows(num_eigs, num_obs - num_eigs));

        //eprintln!("s construction reached");
        let mut pre_s = Vec::with_capacity(num_eigs);
        let mut pre_sqrt_s = Vec::with_capacity(num_eigs);
        for index in 0..num_eigs {
            pre_s.push(pre_epsilon[(index, 0)] * pre_epsilon[(index, 0)] / eps_t_eps_mini);
            pre_sqrt_s.push(num_traits::sign::signum(pre_epsilon[index]) * pre_s[index].sqrt());
        }
        let pre_s_mean_vec = pre_s.clone();

        //eprintln!("pre_eta declaration reached");
        let mut pre_eta = Vec::with_capacity(num_eigs);
        pre_eta.resize(num_eigs, 0.1);

        let mut pre_c_eta = pre_eta.clone();
        pre_c_eta.resize(num_obs, 0.0);

        //eprintln!("pre_eta and pre_c_eta matrix conversion reached");
        let pre_eta = nalgebra::DMatrix::from_vec(num_eigs, 1, pre_eta);
        let pre_c_eta = nalgebra::DMatrix::from_vec(num_obs, 1, pre_c_eta);

        //eprintln!("Construction reached");
        Self {
            num_obs,
            num_eigs,
            scan_index: 1,
            sigma2: 1.0,
            eps_t_eps,
            eps_norm: eps_t_eps.sqrt(),
            epsilon: pre_epsilon,
            kappa: eps_t_eps_mini / eps_t_eps,
            kappa_pp: AdaptiveParams::new(0.0001),
            s: nalgebra::DMatrix::from_vec(num_eigs, 1, pre_s),
            /*
            s_mean_vec: nalgebra::DMatrix::from_vec(
                num_eigs,
                1,
                vec![1.0 / (num_eigs as f64); num_eigs],
            ),
            */
            s_mean_vec: nalgebra::DMatrix::from_vec(num_eigs, 1, pre_s_mean_vec),
            s_alpha: nalgebra::DMatrix::from_vec(num_eigs, 1, vec![1.0; num_eigs]),
            s_pp: AdaptiveParams::new(0.0001),
            sqrt_s: nalgebra::DMatrix::from_vec(num_eigs, 1, pre_sqrt_s),
            //lambda: Lambda::new(num_eigs, 2.0, 2.0, 0.0001),
            lambda: nalgebra::DMatrix::from_vec(num_eigs, 1, vec![1.0; num_eigs]),
            lambda_pp: lambda_pp_vec,
            prior_lambda_shape,
            prior_lambda_rate,
            eta: pre_eta,
            c_eta: pre_c_eta,
        }
    }

    // Implementation of kappa functions
    pub fn sample_kappa(&mut self) {
        //eprintln!("Sampling kappa");
        //eprintln!("Current kappa value: {}", self.kappa);
        //eprintln!("Proposal variance: {}", self.kappa_pp.variance);
        let prop_kappa = self.kappa_pp.sample();
        //eprintln!("Proposal kappa value: {}", prop_kappa);
        if prop_kappa < 0.0 || 1.0 < prop_kappa {
            self.update_kappa(self.kappa);
            return;
        }

        // Calculate difference in log posteriors
        let mut zeta = Vec::with_capacity(self.num_eigs);
        for index in 0..self.num_eigs {
            //zeta[index] = self.lambda[index] / (self.sigma2 + self.lambda[(index,0)]);
            zeta.push(self.lambda[index] / (self.sigma2 + self.lambda[(index, 0)]));
        }
        let zeta = nalgebra::DMatrix::from_vec(self.num_eigs, 1, zeta);
        let llik_prod = self.eps_t_eps * zeta.dot(&self.s) / 2.0 / self.sigma2;
        let llik_diff = (prop_kappa - self.kappa) * llik_prod;
        let prior_diff = (self.num_eigs as f64 / 2.0 - 1.0) * (prop_kappa.ln() - self.kappa.ln())
            + ((self.num_obs - self.num_eigs) as f64 / 2.0 - 1.0)
                * (((1.0 - prop_kappa) as f64).ln() - (1.0 - self.kappa).ln());
        let trans_kern_diff = (prop_kappa + self.kappa - 2.0 * self.kappa_pp.mean)
            * (prop_kappa - self.kappa)
            / 2.0
            / self.kappa_pp.variance;
        let mut rng = rand::thread_rng();
        if llik_diff + prior_diff + trans_kern_diff
            > Uniform::new(0 as f64, 1 as f64).sample(&mut rng).ln()
        {
            self.update_kappa(prop_kappa);
        } else {
            self.update_kappa(self.kappa);
        }
    }
    /*
    pub fn sample_kappa(&mut self) {
        //let mut rng = rand::thread_rng();
        //let pois = Poisson::new(self.kappa * self.eps_t_eps);
    }
    */

    pub fn update_kappa(&mut self, new_kappa: f64) {
        self.kappa = new_kappa;
        self.kappa_pp.update(new_kappa, self.scan_index);
    }

    // Implementation of s functions
    pub fn sample_s(&mut self) {
        //eprintln!("Sampling s");
        //eprintln!("Current s value: {}", self.s);
        let prop_s = self.draw_dirichlet();
        //eprintln!("Proposal s value: {}", prop_s);
        //eprintln!("prop_s: {}", prop_s);
        let mut zeta = Vec::with_capacity(self.num_eigs);
        let mut trans_kern_diff: f64 = 0.0;
        for index in 0..self.num_eigs {
            //zeta[index] = self.lambda[(index, 0)] / (self.sigma2 + self.lambda[(index, 0)]);
            zeta.push(self.lambda[(index, 0)] / (self.sigma2 + self.lambda[(index, 0)]));
            trans_kern_diff += (self.s_alpha[(index, 0)] - 1.0)
                * (self.s[(index, 0)].ln() - prop_s[(index, 0)].ln());
        }
        //eprintln!("self.s: {}", self.s);
        let zeta = nalgebra::DMatrix::from_vec(self.num_eigs, 1, zeta);
        let llik_diff = self.kappa * self.eps_t_eps / 2.0 / self.sigma2
            * (zeta.transpose() * (&prop_s - &self.s))[(0, 0)];
        //* zeta.dot(&(&prop_s - &self.s));
        let prior_diff =
            self.num_eigs as f64 * ((&self.s.dot(&self.s)).ln() - (&prop_s.dot(&prop_s)).ln());
        let mut rng = rand::thread_rng();
        //eprintln!("llik_diff: {}", llik_diff);
        //eprintln!("prior_diff: {}", prior_diff);
        //eprintln!("trans_kern_diff: {}", trans_kern_diff);
        //eprintln!("Log-acceptance probability: {}", llik_diff + prior_diff + trans_kern_diff);
        if llik_diff + prior_diff + trans_kern_diff
            > Uniform::new(0 as f64, 1 as f64).sample(&mut rng).ln()
        {
            //eprintln!("Accepted");
            self.update_s(prop_s);
        } else {
            //eprintln!("Rejected");
            self.noupdate_s();
        }
    }

    pub fn draw_dirichlet(&self) -> nalgebra::DMatrix<f64> {
        let mut dir_vec = nalgebra::DMatrix::from_vec(self.num_eigs, 1, vec![0.0; self.num_eigs]);
        let mut rng = rand::thread_rng();
        let mut element_sum: f64 = 0.0;
        //eprintln!("self.s_alpha: {}", self.s_alpha);
        for index in 0..self.num_eigs {
            //eprintln!("self.s_alpha[({}, 0)]: {}", index, self.s_alpha[(index, 0)]);
            let gamma = Gamma::new(self.s_alpha[(index, 0)], 1.0);
            dir_vec[(index, 0)] = gamma.unwrap().sample(&mut rng);
            element_sum += dir_vec[(index, 0)];
        }
        dir_vec / element_sum
    }

    pub fn update_s(&mut self, new_s: nalgebra::DMatrix<f64>) {
        self.s = new_s;
        for index in 0..self.num_eigs {
            self.sqrt_s[(index, 0)] =
                num_traits::sign::signum(self.sqrt_s[(index, 0)]) * self.s[(index, 0)].sqrt();
        }
        self.s_mean_vec =
            ((self.scan_index - 1) as f64 * &self.s_mean_vec + &self.s) / self.scan_index as f64;
        self.s_pp.update(self.s[(0, 0)], self.scan_index);
        self.s_alpha = (self.s_pp.mean * (1.0 - self.s_pp.mean) / self.s_pp.variance - 1 as f64)
            .abs()
            * &self.s_mean_vec;
    }

    pub fn noupdate_s(&mut self) {
        // Other updates not necessary
        self.s_mean_vec =
            ((self.scan_index - 1) as f64 * &self.s_mean_vec + &self.s) / self.scan_index as f64;
        self.s_pp.update(self.s[(0, 0)], self.scan_index);
        //eprintln!("self.s_pp.mean: {}", self.s_pp.mean);
        //eprintln!("self.s_pp.variance: {}", self.s_pp.variance);
        //eprintln!("self.scan_index: {}", self.scan_index);
        self.s_alpha = (self.s_pp.mean * (1.0 - self.s_pp.mean) / self.s_pp.variance - 1 as f64)
            .abs()
            * &self.s_mean_vec;
    }

    // Implementation of lambda functions
    pub fn update_lambda_i(&mut self, new_lambda_i: f64, index: usize) {
        self.lambda[(index, 0)] = new_lambda_i;
        self.lambda_pp[index].update(new_lambda_i, self.scan_index);
    }

    pub fn sample_lambda(&mut self) {
        //eprintln!("Sampling lambda");
        //eprintln!("Current lambda value: {}", self.lambda);
        for index in 0..self.num_eigs {
            let prop_lambda_i = self.lambda_pp[index].sample();
            if prop_lambda_i < 0.0 {
                self.update_lambda_i(self.lambda[(index, 0)], index);
                continue;
            }

            let curr_lambda_i = self.lambda[(index, 0)];
            let curr_l_post = -(self.sigma2 + curr_lambda_i).ln() / 2.0
                + self.kappa * self.eps_t_eps * curr_lambda_i * self.s[(index, 0)]
                    / (self.sigma2 + curr_lambda_i)
                    / 2.0
                    / self.sigma2;
            let prop_l_post = -(self.sigma2 + prop_lambda_i).ln() / 2.0
                + self.kappa * self.eps_t_eps * prop_lambda_i * self.s[(index, 0)]
                    / (self.sigma2 + prop_lambda_i)
                    / 2.0
                    / self.sigma2;
            let prior_diff = (self.prior_lambda_shape - 1.0)
                * (prop_lambda_i.ln() - curr_lambda_i.ln())
                - self.prior_lambda_rate * (prop_lambda_i - curr_lambda_i);
            let trans_kern_diff = (prop_lambda_i + curr_lambda_i
                - 2.0 * self.lambda_pp[index].mean)
                * (prop_lambda_i - curr_lambda_i)
                / 2.0
                / self.lambda_pp[index].variance;
            let mut rng = rand::thread_rng();
            if prop_l_post - curr_l_post + prior_diff + trans_kern_diff
                > Uniform::new(0 as f64, 1 as f64).sample(&mut rng).ln()
            {
                self.update_lambda_i(prop_lambda_i, index);
            } else {
                self.update_lambda_i(curr_lambda_i, index);
            }
        }
    }

    // greObj specific functions - already implemented, but need to be altered
    pub fn update_epsilon_kappa_and_s(&mut self, new_epsilon: Vec<f64>) {
        // Going to be a beast
        // Replace this
        // let new_epsilon: nalgebra::DMatrix<f64> = nalgebra::DMatrix::from_vec(self.num_obs, 1, Vec::with_capacity(self.num_obs));
        // With this
        //eprintln!("update_epsilon_kappa_and_s entered");
        //eprintln!("Old kappa: {}", self.kappa);
        //eprintln!("Old s: {}", self.s);
        let new_epsilon: nalgebra::DMatrix<f64> =
            nalgebra::DMatrix::from_vec(self.num_obs, 1, new_epsilon);
        //eprintln!("new_epsilon successfully assigned");
        // End of replace this section
        let mut omega: Vec<f64> = Vec::with_capacity(self.num_eigs);
        omega.resize(self.num_eigs, 0.0);
        //let mut omega = nalgebra::DMatrix::from_vec(self.num_eigs, 1, omega); // We'll do this later
        let mut rng = rand::thread_rng();
        /*
        let mut new_sqrt_s: Vec<f64> = Vec::with_capacity(self.num_eigs);
        new_sqrt_s.resize(self.num_eigs, 0.0);
        for index in 0..self.num_eigs {
            new_sqrt_s[index] = self.sqrt_s[(index, 0)];
        }
        let new_sqrt_s = nalgebra::DMatrix::from_vec(self.num_eigs, 1, new_sqrt_s);
        //eprintln!("new_sqrt_s: {}", new_sqrt_s);
        */

        //eprintln!("new_epsilon: {}", new_epsilon);
        //eprintln!("self.epsilon: {}", self.epsilon);
        //eprintln!("self.c_eta: {}", self.c_eta);

        let ne_t_ne: f64 = (&new_epsilon).dot(&new_epsilon);
        let eps_t_ne = (&self.epsilon).dot(&new_epsilon);
        let eta_t_eta = (&self.eta).dot(&self.eta);
        let beta_t_ne = (&self.c_eta).dot(&new_epsilon);
        let sqrt_s_t_eta = (&self.sqrt_s).dot(&self.eta);

        let mut eps_sub_term: f64 = self.eps_t_eps;
        let mut beta_sub_term: f64 = eta_t_eta;
        let mut cross_sub_term: f64 = self.eps_norm * self.kappa.sqrt() * sqrt_s_t_eta;
        let mut eps_t_ne_sub_term: f64 = eps_t_ne;
        let mut beta_t_ne_sub_term: f64 = beta_t_ne;
        let mut omega_t_omega: f64 = 0.0;

        assert!(
            !beta_t_ne_sub_term.is_nan(),
            "beta_t_ne: {}
            self.c_eta: {}
            new_epsilon: {}",
            beta_t_ne,
            self.c_eta,
            new_epsilon
        );

        let mut tau_t_tau: f64 = (eta_t_eta * self.kappa * self.s[(0, 0)]
            - 2.0
                * self.kappa
                * &self.eta.dot(&self.sqrt_s)
                * &self.eta[(0, 0)]
                * &self.sqrt_s[(0, 0)]
            + self.eta[(0, 0)].pow(2))
            / (eta_t_eta - self.kappa * sqrt_s_t_eta.pow(2));
        //tau_t_tau = std::cmp::min(tau_t_tau, 1.0); // TODO: Assert something
        let beta = Beta::new(0.5, ((self.num_obs - 3) as f64) / 2.0).unwrap();
        let mut cos_phi = beta.sample(&mut rng).sqrt();

        let mut denom: f64 = eps_sub_term * beta_sub_term - cross_sub_term.pow(2);

        let mut eps_tilde_norm: f64 = (ne_t_ne
            - (beta_sub_term * eps_t_ne_sub_term * eps_t_ne_sub_term
                - 2.0 * cross_sub_term * eps_t_ne_sub_term * beta_t_ne_sub_term
                + eps_sub_term * beta_t_ne_sub_term * beta_t_ne_sub_term)
                / denom)
            .sqrt();
        assert!(
            ne_t_ne
                - (beta_sub_term * eps_t_ne_sub_term * eps_t_ne_sub_term
                    - 2.0 * cross_sub_term * eps_t_ne_sub_term * beta_t_ne_sub_term
                    + eps_sub_term * beta_t_ne_sub_term * beta_t_ne_sub_term)
                    / denom
                > 0.0,
            "ne_t_ne: {}
            beta_sub_term: {}
            eps_t_ne_sub_term: {}
            cross_sub_term: {}
            beta_t_ne_sub_term: {}
            combination: {}",
            ne_t_ne,
            beta_sub_term,
            eps_t_ne_sub_term,
            cross_sub_term,
            beta_t_ne_sub_term,
            (beta_sub_term * eps_t_ne_sub_term * eps_t_ne_sub_term
                - 2.0 * cross_sub_term * eps_t_ne_sub_term * beta_t_ne_sub_term
                + eps_sub_term * beta_t_ne_sub_term * beta_t_ne_sub_term)
                / denom
        );
        /*
        eprintln!("eps_tilde_norm: {}", eps_tilde_norm);
        eprintln!("ne_t_ne: {}", ne_t_ne);
        eprintln!("denom: {}", denom);
        eprintln!("self.sqrt_s: {}", self.sqrt_s);
        eprintln!("eta_t_eta: {}", eta_t_eta);
        eprintln!("self.eps_norm: {}", self.eps_norm);
        eprintln!("sqrt_s_t_eta: {}", sqrt_s_t_eta);
        eprintln!("sqrt_s_t_eta.pow(2): {}", sqrt_s_t_eta.pow(2));
        eprintln!("beta_t_ne: {}", beta_t_ne);
        eprintln!("cos_phi: {}", cos_phi);
        eprintln!("tau_t_tau: {}", tau_t_tau);
        */

        omega[0] = (eta_t_eta * self.eps_norm * self.kappa.sqrt() * self.sqrt_s[(0, 0)]
            - self.eps_norm * self.kappa.sqrt() * sqrt_s_t_eta * self.eta[(0, 0)] * eps_t_ne)
            / (self.eps_t_eps * eta_t_eta - sqrt_s_t_eta.pow(2))
            + ((self.eps_t_eps * self.eta[(0, 0)]
                - self.eps_t_eps
                    * self.kappa
                    * sqrt_s_t_eta
                    * self.eta[(0, 0)]
                    * self.sqrt_s[(0, 0)])
                * beta_t_ne)
                / (self.eps_t_eps * eta_t_eta - sqrt_s_t_eta.pow(2))
            + (1.0 - tau_t_tau).sqrt() * eps_tilde_norm * cos_phi;

        //eprintln!("omega[0]: {}", omega[0]);

        for index in 1..(self.num_eigs - 1) {
            eps_sub_term -= self.eps_t_eps * self.kappa * self.s[(index - 1, 0)];
            beta_sub_term -= self.eta[(index - 1, 0)].pow(2);
            cross_sub_term -= self.eps_norm
                * self.kappa.sqrt()
                * self.sqrt_s[(index - 1, 0)]
                * self.eta[(index - 1, 0)];
            eps_t_ne_sub_term -=
                omega[index - 1] * self.eps_norm * self.kappa.sqrt() * self.sqrt_s[(index - 1, 0)];
            beta_t_ne_sub_term -= omega[index - 1] * self.eta[(index - 1, 0)];
            omega_t_omega += omega[index - 1].pow(2);

            denom = eps_sub_term * beta_sub_term - cross_sub_term.pow(2);

            tau_t_tau = (beta_sub_term * self.eps_t_eps * self.kappa * self.s[(index, 0)]
                - 2.0
                    * cross_sub_term
                    * self.eps_norm
                    * self.kappa.sqrt()
                    * self.sqrt_s[(index, 0)]
                    * self.eta[(index, 0)]
                + eps_sub_term * self.eta[(index, 0)].pow(2))
                / denom;

            let beta = Beta::new(0.5, ((self.num_obs - index - 3) as f64) / 2.0).unwrap();
            cos_phi = beta.sample(&mut rng).sqrt();

            eps_tilde_norm = (ne_t_ne
                - omega_t_omega
                - (beta_sub_term * eps_t_ne_sub_term * eps_t_ne_sub_term
                    - 2.0 * cross_sub_term * eps_t_ne_sub_term * beta_t_ne_sub_term
                    + eps_sub_term * beta_t_ne_sub_term * beta_t_ne_sub_term)
                    / denom)
                .sqrt();

            omega[index] = (beta_sub_term
                * self.eps_norm
                * self.kappa.sqrt()
                * self.sqrt_s[(index, 0)]
                - cross_sub_term * self.eta[(index, 0)])
                * eps_t_ne_sub_term
                / denom
                + (eps_sub_term * self.eta[(index, 0)]
                    - cross_sub_term * self.eps_norm * self.kappa.sqrt() * self.sqrt_s[(index, 0)])
                    * beta_t_ne_sub_term
                    / denom
                + (1.0 - tau_t_tau).sqrt() * eps_tilde_norm * cos_phi;

            //eprintln!("omega[index]: {}", omega[index]);
        }

        eps_sub_term -= self.eps_t_eps * self.kappa * self.s[(self.num_eigs - 2, 0)];
        beta_sub_term -= self.eta[(self.num_eigs - 2, 0)].pow(2);
        cross_sub_term -= self.eps_norm
            * self.kappa.sqrt()
            * self.sqrt_s[(self.num_eigs - 2, 0)]
            * self.eta[(self.num_eigs - 2, 0)];
        eps_t_ne_sub_term -= omega[self.num_eigs - 2]
            * self.eps_norm
            * self.kappa.sqrt()
            * self.sqrt_s[(self.num_eigs - 2, 0)];
        beta_t_ne_sub_term -= omega[self.num_eigs - 2] * self.eta[(self.num_eigs - 2, 0)];
        omega_t_omega += omega[self.num_eigs - 2].pow(2);

        denom = eps_sub_term * beta_sub_term - cross_sub_term.pow(2);

        omega[self.num_eigs - 1] = (beta_sub_term
            * self.eps_norm
            * self.kappa.sqrt()
            * self.sqrt_s[(self.num_eigs - 1, 0)]
            - cross_sub_term * self.eta[(self.num_eigs - 1, 0)])
            * eps_t_ne_sub_term
            / denom
            + (eps_sub_term * self.eta[(self.num_eigs - 1, 0)]
                - cross_sub_term
                    * self.eps_norm
                    * self.kappa.sqrt()
                    * self.sqrt_s[(self.num_eigs - 1, 0)])
                * beta_t_ne_sub_term
                / denom;

        let omega = nalgebra::DMatrix::from_vec(self.num_eigs, 1, omega);
        //eprintln!("omega: {}", omega);

        // Updating the quantities that really matter
        self.kappa = omega_t_omega / ne_t_ne;
        self.sqrt_s = omega / omega_t_omega.sqrt();
        for index in 0..self.num_eigs {
            self.s[(index, 0)] = self.sqrt_s[(index, 0)].pow(2);
        }
        self.epsilon = new_epsilon;
        self.eps_t_eps = ne_t_ne;
        self.eps_norm = ne_t_ne.sqrt();
        //eprintln!("New kappa: {}", self.kappa);
        //eprintln!("New s: {}", self.s);
    }

    //#[roxido]
    pub fn update_sigma2(&mut self, resid_var: f64) {
        self.sigma2 = resid_var;
    }

    pub fn run_sampler(&mut self) {
        self.sample_kappa();
        self.sample_s();
        self.sample_lambda();
        self.sample_eta();
        self.sample_c_eta();
        self.scan_index += 1;
    }

    pub fn sample_eta(&mut self) {
        let mut rng = rand::thread_rng();
        for index in 0..self.num_eigs {
            // The name variance is a little misleading - needs to be scaled by self.sigma2
            let variance = self.lambda[(index, 0)] / (self.sigma2 + self.lambda[(index, 0)]);
            let mean = self.kappa.sqrt() * self.eps_norm * variance * self.sqrt_s[(index, 0)];
            let norm = Normal::new(mean, self.sigma2.sqrt() * variance.sqrt()).unwrap();
            self.eta[(index, 0)] = norm.sample(&mut rng);
        }
    }

    pub fn sample_c_eta(&mut self) {
        let eta_norm = (&self.eta.transpose() * &self.eta)[(0, 0)].sqrt();
        //let eta_norm = (&self.eta).dot(&self.eta).sqrt();
        //eprintln!("In sample_c_eta: eta_norm: {}", eta_norm);
        //let sqrt_s = self.s.clone(); // TODO: This is wrong, or at least incomplete
        /*
        let mut sqrt_s: Vec<f64> = Vec::with_capacity(self.num_eigs);
        for index in 0..self.num_eigs {
            sqrt_s.push(self.s[(index, 0)].sqrt());
        }
        let sqrt_s = nalgebra::DMatrix::from_vec(self.num_eigs, 1, sqrt_s);
        */
        let cos_phi = self.kappa.sqrt() * (&self.sqrt_s.transpose() * &self.eta)[(0, 0)] / eta_norm;
        //let cos_phi = self.kappa.sqrt() * sqrt_s.dot(&self.eta);
        //eprintln!("In sample_c_eta: cos_phi: {}", cos_phi);
        let center = &self.epsilon * cos_phi * eta_norm / self.eps_norm;
        //eprintln!("In sample_c_eta: self.epsilon: {}", self.epsilon);
        //eprintln!("In sample_c_eta: center: {}", center);
        //eprintln!("In sample_c_eta: self.sum_of_squared_epsilons {}", self.sum_of_squared_epsilons);
        //let mut rng = rand::thread_rng();
        //let std_norm = StandardNormal::new().unwrap();
        let mut z: Vec<f64> = Vec::with_capacity(self.num_obs);
        for _index in 0..self.num_obs {
            //z.push(std_norm.sample(&mut rng));
            z.push(rand::thread_rng().sample(StandardNormal));
        }
        let z = nalgebra::DMatrix::from_vec(self.num_obs, 1, z);
        let orth_proj = &z - &self.epsilon * (&self.epsilon.transpose() * &z) / self.eps_t_eps;
        //let orth_proj = &z - &self.epsilon * (&self.epsilon).dot(&z) / self.eps_t_eps;
        let orth_proj = &orth_proj / (&orth_proj.transpose() * &orth_proj)[(0, 0)].sqrt();
        //let orth_proj = &orth_proj / (&orth_proj).dot(&orth_proj).sqrt();
        self.c_eta = &center + &orth_proj * eta_norm * (1.0 - cos_phi * cos_phi).sqrt();
        // Borrows of center and orth_proj can be removed after testing
        // Maybe don't have to be?
        assert!(
            !self.c_eta[(0, 0)].is_nan(),
            "center: {}
            orth_proj: {},
            eta_norm: {}
            cos_phi: {}",
            center,
            orth_proj,
            eta_norm,
            cos_phi
        )
        //eprintln!("c_eta: {}", self.c_eta);
    }

    pub fn get_c_eta(&self) -> &nalgebra::DMatrix<f64> {
        // I'm not sure if this lets memory be altered in R - I hope not
        &self.c_eta
    }
}
