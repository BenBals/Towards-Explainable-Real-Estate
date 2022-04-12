use algorithms::sweepline;
use std::convert::TryInto;
use test_helpers::Point;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

const POINT_MAX: u64 = 1000 * 10; // km * units/km (accuracy of 100m)

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

fn sweepline_bench(n: u64, radius: u64) {
    let points = get_random_points(n);
    let mut graph = vec![];

    sweepline::for_every_close_point_do(&points, radius, |_key, neighbours| {
        graph.push(neighbours.len())
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sweepline_random_points");
    group.bench_function("sweepline_n_1000_radius_100m", |b| {
        b.iter(|| sweepline_bench(black_box(1000), black_box(10)))
    });
    group.sample_size(10);
    group.bench_function("random_point_1000000", |b| {
        b.iter(|| get_random_points(black_box(1000000)).len())
    });
    group.bench_function("sweepline_n_1000000_radius_10km", |b| {
        b.iter(|| sweepline_bench(black_box(1000000), black_box(10 * 10)))
    });
    group.bench_function("sweepline_n_1000000_radius_100km", |b| {
        b.iter(|| sweepline_bench(black_box(1000000), black_box(100 * 10)))
    });
    group.finish()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
