use std::sync::atomic::AtomicUsize;

use algorithms::QuadraticApproximation;
use common::{database, BpResult};
use executables::filter_from_path_or_panic;
use itertools::Itertools;
use rayon::prelude::*;

fn main() -> BpResult<()> {
    common::logging::init_logging();

    let filter = filter_from_path_or_panic("../queries/berlin.json");
    let immos = database::read_reasonable_immos(None, Some(filter), None)?;
    log::info!("Got {} immos", immos.len(),);

    let count = AtomicUsize::new(0);

    immos.par_iter().for_each(|middle_immo| {
        let distances: Vec<_> = immos
            .iter()
            .filter(|&new_immo| new_immo != middle_immo)
            .map(|immo| middle_immo.plane_distance(immo).unwrap() as u64 + 1)
            .sorted()
            .map(|integer| integer as f64)
            .collect();

        QuadraticApproximation::from_distances_with_overestimation_threshold(distances, f64::NAN);
        let last_count = count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        log::info!("Done {} immos", last_count + 1);
    });

    Ok(())
}
