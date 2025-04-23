// The 'roxido_registration' macro is called at the start of the 'lib.rs' file.
roxido_registration!();
//use rbindings::GetRNGstate;
//use rbindings::PutRNGstate;
//use rbindings::Rf_unprotect;
use roxido::*;

use crate::gre::GreObj;

mod adaptive_params;
//mod kappa;
//mod s;
//mod lambda;
mod gre;
//mod gre_new;

#[roxido]
fn initialize_gre(n: usize, m: usize, prior_lambda_shape: f64, prior_lambda_rate: f64) {
    //unsafe {
    // The unsafe block appears to be necessary to get/put the RNG state
    //GetRNGstate();
    let mut gre = GreObj::new(
        n as usize,
        m as usize,
        prior_lambda_shape,
        prior_lambda_rate,
    );
    // Eta can't be initialized to zero or we get runtime errors
    gre.sample_eta();
    gre.sample_c_eta();
    //PutRNGstate();
    RExternalPtr::encode(gre, "gre", pc)
    //}
}

#[roxido]
fn update_epsilon(gre_ptr: &mut RExternalPtr, new_eps: &RVector) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    let epsilon = new_eps.to_f64(pc).slice().to_vec();
    let n = epsilon.len();
    if n != gre.num_obs {
        //return Err("."); // Not sure how this works - will have to be careful
        // Accounted for in R code
    }
    //unsafe {
    //GetRNGstate();
    gre.update_epsilon_kappa_and_s(epsilon);
    //PutRNGstate();
    //Rf_unprotect(1);
    //}
}

#[roxido]
fn update_sigma2(gre_ptr: &mut RExternalPtr, new_sigma2: f64) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    //unsafe {
    //GetRNGstate();
    gre.update_sigma2(new_sigma2);
    //PutRNGstate();
    //Rf_unprotect(1);
    //}
}

#[roxido]
fn run(gre_ptr: &mut RExternalPtr) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    //unsafe {
    //GetRNGstate();
    gre.run_sampler();
    //PutRNGstate();
    //Rf_unprotect(1);
    //}
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

// The following functions are just for testing purposes
#[roxido]
fn get_kappa(gre_ptr: &mut RExternalPtr) {
    let gre: &GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    let kappa = gre.get_kappa();
    kappa
}

#[roxido]
fn get_s(gre_ptr: &mut RExternalPtr) {
    let gre: &GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    let s = gre.get_s();
    let ret = RVector::from_value(0.0, gre.num_eigs, pc);
    for index in 0..gre.num_eigs {
        ret.set(index, s[index]).stop();
    }
    ret
}

#[roxido]
fn get_lambda(gre_ptr: &mut RExternalPtr) {
    let gre: &GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    let lambda = gre.get_lambda();
    let ret = RVector::from_value(0.0, gre.num_eigs, pc);
    for index in 0..gre.num_eigs {
        ret.set(index, lambda[index]).stop();
    }
    ret
}

#[roxido]
fn get_eta(gre_ptr: &mut RExternalPtr) {
    let gre: &GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    let eta = gre.get_eta();
    let ret = RVector::from_value(0.0, gre.num_eigs, pc);
    for index in 0..gre.num_eigs {
        ret.set(index, eta[index]).stop();
    }
    ret
}

#[roxido]
fn strongarm_update_everything(
    gre_ptr: &mut RExternalPtr,
    new_kappa: &RScalar,
    new_s: &RVector,
    new_lambda: &RVector,
    new_eta: &RVector,
    new_c_eta: &RVector,
) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    let kappa = new_kappa.f64();
    let s = new_s.to_f64(pc).slice().to_vec();
    let lambda = new_lambda.to_f64(pc).slice().to_vec();
    let eta = new_eta.to_f64(pc).slice().to_vec();
    let c_eta = new_c_eta.to_f64(pc).slice().to_vec();
    gre.strongarm_update_everything(kappa, s, lambda, eta, c_eta);
}

#[roxido]
fn sample_kappa(gre_ptr: &mut RExternalPtr) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    gre.sample_kappa();
}
#[roxido]
fn sample_s(gre_ptr: &mut RExternalPtr) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    gre.sample_s();
}
#[roxido]
fn sample_lambda(gre_ptr: &mut RExternalPtr) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    gre.sample_lambda();
}
#[roxido]
fn sample_eta(gre_ptr: &mut RExternalPtr) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    gre.sample_eta();
}
#[roxido]
fn sample_c_eta(gre_ptr: &mut RExternalPtr) {
    let gre: &mut GreObj = RExternalPtr::decode_mut::<GreObj>(gre_ptr);
    gre.sample_c_eta();
}
