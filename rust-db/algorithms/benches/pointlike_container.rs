use algorithms::segment_tree::PointlikeContainer;
use std::convert::TryInto;
use test_helpers::Point;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

// This Benchmark simulates Berlin
const UNITS_PER_KM: u64 = 1000; // km * units/km (accuracy of 10m)
const POINT_MAX: u64 = 30 * UNITS_PER_KM; // km * units/km (accuracy of 10m)
const NUM_POINTS: u64 = 20_000;
const NUM_QUERIES: u64 = NUM_POINTS;

const RNG_SEED: [u8; 16] = *b"0123456789abcdef";

fn get_random_points(n: u64) -> Vec<Point> {
    let mut rng = XorShiftRng::from_seed(RNG_SEED);

    (0..n)
        .map(|i| {
            let x = rng.gen_range(0..POINT_MAX);
            let y = rng.gen_range(0..POINT_MAX);
            Point::new(
                x,
                y,
                i.try_into().expect("failed conversion from usize to u64"),
            )
        })
        .collect()
}

fn creation_bench(points: &[Point]) -> PointlikeContainer<Point> {
    PointlikeContainer::with(points.iter().copied()).unwrap()
}

fn query_bench(num_queries: u64, range: u64, container: &PointlikeContainer<Point>) {
    let mut rng = XorShiftRng::from_seed(RNG_SEED);
    for _ in 0..num_queries {
        let x_left = rng.gen_range(0..POINT_MAX);
        let y_left = rng.gen_range(0..POINT_MAX);
        let x_right = x_left + range;
        let y_right = y_left + range;

        container.for_each_in_range(x_left..x_right, y_left..y_right, |f| {
            black_box(f);
        });
    }
}

fn parallel_query_bench(num_queries: u64, range: u64, container: &PointlikeContainer<Point>) {
    use rayon::prelude::*;

    let mut rng = XorShiftRng::from_seed(RNG_SEED);
    let range: Vec<_> = (0..num_queries)
        .map(|_| {
            let x_left = rng.gen_range(0..POINT_MAX);
            let y_left = rng.gen_range(0..POINT_MAX);
            let x_right = x_left + range;
            let y_right = y_left + range;
            (x_left..x_right, y_left..y_right)
        })
        .collect();

    range.into_par_iter().for_each(|(x_range, y_range)| {
        container.for_each_in_range(x_range, y_range, |p| {
            black_box(p);
        });
    });
}

fn creation_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("pointlike_container_random_points_creation");

    group.sample_size(25);
    group.bench_function(format!("{}_points", NUM_POINTS), |b| {
        let points = get_random_points(NUM_POINTS);
        b.iter_with_large_drop(|| creation_bench(black_box(&points)));
    });
    group.finish();
}

fn query_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("pointlike_container_random_points_query");

    let points = get_random_points(NUM_POINTS);
    let container = PointlikeContainer::with(points).unwrap();

    group.bench_function("5km", |b| {
        b.iter(|| query_bench(1, 5 * UNITS_PER_KM, &container));
    });

    group.bench_function(format!("5km_{}_times", NUM_QUERIES), |b| {
        b.iter(|| query_bench(NUM_QUERIES, 5 * UNITS_PER_KM, &container));
    });

    group.bench_function(format!("5km_{}_times_parallel", NUM_QUERIES), |b| {
        b.iter(|| parallel_query_bench(NUM_QUERIES, 5 * UNITS_PER_KM, &container));
    });

    group.bench_function("300m", |b| {
        b.iter(|| query_bench(1, (0.3 * UNITS_PER_KM as f64) as u64, &container));
    });

    group.bench_function(format!("300m_{}_times", NUM_QUERIES), |b| {
        b.iter(|| query_bench(NUM_QUERIES, (0.3 * UNITS_PER_KM as f64) as u64, &container));
    });

    group.bench_function(format!("300m_{}_times_parallel", NUM_QUERIES), |b| {
        b.iter(|| {
            parallel_query_bench(NUM_QUERIES, (0.3 * UNITS_PER_KM as f64) as u64, &container)
        });
    });

    group.finish()
}

criterion_group!(creation, creation_group);
criterion_group!(query, query_group);
criterion_main!(creation, query);
