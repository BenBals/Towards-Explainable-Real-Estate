use algorithms::segment_tree::IntervalTree;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::cmp::min;
use std::convert::TryInto;

// Germany is only 1000km large. This will simulate accuracy of 10m
const TREE_MAX: u64 = 1000 * 100; // km * units/km

// how big will the boxes you search for be?
const SEARCH_WIDTH: u64 = 100 * 100; // km * units/km

const RNG_SEED: [u8; 16] = *b"0123456789abcdef";

/// generate n random points on the natural number line segment [0, TREE_MAX)
fn get_random_points(n: u64) -> Vec<(u64, u64)> {
    let mut rng = XorShiftRng::from_seed(RNG_SEED);

    (0..n)
        .map(|idx| {
            let coord = rng.gen_range(0..TREE_MAX);
            (coord, idx)
        })
        .collect()
}

fn add_query_delete_n(n: u64) {
    let mut tree: IntervalTree<u64> = IntervalTree::new(0..TREE_MAX);

    let intervals = get_random_points(n);

    intervals
        .chunks((TREE_MAX / SEARCH_WIDTH + 1).try_into().unwrap())
        .for_each(|chunk| {
            chunk.iter().for_each(|(point, value)| {
                tree.add(*point..min(*point + SEARCH_WIDTH, TREE_MAX), *value)
                    .unwrap_or_else(|_| panic!("We only perform vaild adds in the benchmark."));
            });

            chunk.iter().for_each(|(point, _value)| {
                tree.query(*point).count();
            });

            chunk.iter().for_each(|(_point, value)| {
                tree.delete(value)
                    .unwrap_or_else(|_| panic!("We only perform valid deletion in the benchmark."));
            });
        })
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("interval_tree_add_query_delete");
    group.bench_function("interval_tree_1000", |b| {
        b.iter(|| add_query_delete_n(black_box(1000)))
    });
    group.sample_size(10);
    group.bench_function("random_point_1000000", |b| {
        b.iter(|| get_random_points(black_box(1000000)).len())
    });
    group.bench_function("interval_tree_1000000", |b| {
        b.iter(|| add_query_delete_n(black_box(1000000)))
    });
    group.finish()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
