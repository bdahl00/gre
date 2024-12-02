use rand_distr::{Distribution, Normal};

#[derive(Clone)]
pub struct AdaptiveParams {
    pub mean: f64,
    pub variance: f64,
    epsilon: f64,
}

impl AdaptiveParams {
    pub fn new(epsilon: f64) -> Self {
        Self{mean: 0.0, variance: 1.0, epsilon}
    }

    pub fn update(&mut self, draw: f64, scan_index: usize) {
        if scan_index > 0 {
            let new_mean = ((scan_index - 1) as f64 * self.mean + draw) / scan_index as f64;

            if scan_index == 1 {
                self.variance = self.epsilon;
            } else {
                let new_var = (scan_index - 2) as f64 * (self.variance - self.epsilon) + (scan_index - 1) as f64 * self.mean * self.mean + draw * draw;
                let new_var = (new_var - scan_index as f64 * new_mean * new_mean) / (scan_index - 1) as f64;
                self.variance = new_var + self.epsilon;
            }
            self.mean = new_mean;
        } else {
            self.mean = 0.0;
            self.variance = self.epsilon;
        }
    }

    pub fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        let norm = Normal::new(self.mean, self.variance.sqrt(),).unwrap();
        norm.sample(&mut rng)
    }
}
