// The 'roxido_registration' macro is called at the start of the 'lib.rs' file.
roxido_registration!();
use roxido::*;

use crate::gre::GreObj;

mod adaptive_params;
//mod kappa;
//mod s;
//mod lambda;
mod gre;

#[roxido]
fn initialize_gre(n: usize, m: usize, prior_lambda_shape: f64, prior_lambda_rate: f64) {
    let mut gre = GreObj::new(
        n as usize,
        m as usize,
        prior_lambda_shape,
        prior_lambda_rate,
    );
    // Eta can't be initialized to zero or we get runtime errors
    gre.sample_eta();
    gre.sample_c_eta();
    RExternalPtr::encode(gre, "gre", pc)
}

#[roxido]
fn update_epsilon(gre_ptr: &mut RExternalPtr, new_eps: &RVector) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    let epsilon = new_eps.to_f64(pc).slice().to_vec();
    let n = epsilon.len();
    if n != gre.num_obs {
        //return Err("."); // Not sure how this works - will have to be careful
    }
    gre.update_epsilon_kappa_and_s(epsilon);
}

#[roxido]
fn update_sigma2(gre_ptr: &mut RExternalPtr, new_sigma2: f64) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    gre.update_sigma2(new_sigma2);
}

#[roxido]
fn run(gre_ptr: &mut RExternalPtr) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    gre.run_sampler();
}

#[roxido]
fn get_c_eta(gre_ptr: &mut RExternalPtr) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    let c_eta = gre.get_c_eta();
    let ret = RVector::from_value(0.0, gre.num_obs, pc);
    for index in 0..gre.num_obs {
        ret.set(index, c_eta[(index, 0)]).stop();
    }
    ret
}
