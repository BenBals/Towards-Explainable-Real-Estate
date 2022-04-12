use algorithms::segment_tree::PointlikeContainer;
use common::Pointlike;
use kd_tree::KdTree2;
use rand_xorshift::XorShiftRng;
use rstar::{primitives::PointWithData, PointDistance, RTree};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};

// This Benchmark simulates Berlin
const UNITS_PER_KM: i64 = 1000; // km * units/km (accuracy of 1m)
const POINT_MAX: i64 = 30 * UNITS_PER_KM; // 30km
const NUM_POINTS: usize = 20_000;

const RNG_SEED: [u8; 16] = *b"0123456789abcdef";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Point([i64; 2]);

impl Point {
    pub fn distance_2(&self, other: &Self) -> i64 {
        let dx = self.0[0] - other.0[0];
        let dy = self.0[1] - other.0[1];
        dx * dx + dy * dy
    }
}

type RStarPoint = PointWithData<(), [i64; 2]>;

impl From<RStarPoint> for Point {
    fn from(point: RStarPoint) -> Self {
        Self([point.position()[0], point.position()[1]])
    }
}

impl From<Point> for RStarPoint {
    fn from(seg_tree_point: Point) -> Self {
        RStarPoint::new((), [seg_tree_point.0[0], seg_tree_point.0[1]])
    }
}

type KdTreePoint = [i64; 2];

impl From<Point> for KdTreePoint {
    fn from(point: Point) -> Self {
        point.0
    }
}

impl From<KdTreePoint> for Point {
    fn from(point: KdTreePoint) -> Self {
        Self(point)
    }
}

impl Pointlike for Point {
    fn x(&self) -> u64 {
        self.0[0] as u64
    }

    fn y(&self) -> u64 {
        self.0[1] as u64
    }
}

fn random_point<R: Rng>(rng: &mut R) -> Point {
    let x = rng.gen_range(0..POINT_MAX);
    let y = rng.gen_range(0..POINT_MAX);
    Point([x, y])
}

fn get_random_points<R: Rng>(n: usize, rng: &mut R) -> Vec<Point> {
    (0..n).map(|_| random_point(rng)).collect()
}

fn get_random_rstar_points<R: Rng>(n: usize, rng: &mut R) -> Vec<RStarPoint> {
    (0..n).map(|_| random_point(rng).into()).collect()
}

fn get_rtrees<R: Rng>(num_trees: usize, num_points: usize, rng: &mut R) -> Vec<RTree<RStarPoint>> {
    (0..num_trees)
        .map(|_| {
            let points = get_random_rstar_points(num_points, rng);
            RTree::bulk_load(points)
        })
        .collect()
}

fn get_kd_trees<R: Rng>(
    num_trees: usize,
    num_points: usize,
    rng: &mut R,
) -> Vec<KdTree2<KdTreePoint>> {
    (0..num_trees)
        .map(|_| {
            let points: Vec<KdTreePoint> = get_random_points(num_points, rng)
                .into_iter()
                .map(Into::into)
                .collect();
            KdTree2::build(points)
        })
        .collect()
}

fn get_seg_trees<R: Rng>(
    num_trees: usize,
    num_points: usize,
    rng: &mut R,
) -> Vec<PointlikeContainer<Point>> {
    (0..num_trees)
        .map(|_| {
            let points = get_random_points(num_points, rng);
            PointlikeContainer::with(points).unwrap()
        })
        .collect()
}

fn get_random_query_with_count<R: Rng>(neighbours: usize, rng: &mut R) -> (Point, usize) {
    (random_point(rng), neighbours)
}

fn get_random_query_with_distance_2<R: Rng>(
    neighbours: usize,
    tree: &RTree<RStarPoint>,
    rng: &mut R,
) -> (Point, i64) {
    let point = random_point(rng);
    let rstar_point: RStarPoint = point.into();
    (
        point,
        tree.nearest_neighbor_iter_with_distance_2(rstar_point.position())
            .take(neighbours)
            .last()
            .unwrap()
            .1,
    )
}

fn rstar_query_bench_nearest(query: (Point, usize), container: &RTree<RStarPoint>) -> usize {
    let (point, count) = query;
    let point: RStarPoint = point.into();
    black_box(
        container
            .nearest_neighbor_iter(point.position())
            .take(count),
    )
    .count()
}

fn rstar_query_bench_within_distance(query: (Point, i64), container: &RTree<RStarPoint>) -> usize {
    let (point, distance) = query;
    let point: RStarPoint = point.into();
    black_box(container.locate_within_distance(*point.position(), distance)).count()
}

fn rstar_query_bench_within_distance_exact(
    query: (Point, i64),
    count: usize,
    container: &RTree<RStarPoint>,
) -> usize {
    let (point, distance) = query;
    let point: RStarPoint = point.into();
    let mut all: Vec<_> = container
        .locate_within_distance(*point.position(), distance)
        .collect();
    all.select_nth_unstable_by_key(count - 1, |other| {
        point.position().distance_2(other.position())
    });
    black_box(all.len())
}

fn kd_tree_query_bench_nearest(query: (Point, usize), container: &KdTree2<KdTreePoint>) -> usize {
    let (point, count) = query;
    let point: KdTreePoint = point.into();
    let all = container.nearests(&point, count);
    all.len()
}

fn kd_tree_query_bench_within_distance(
    query: (Point, f64),
    container: &KdTree2<KdTreePoint>,
) -> usize {
    let (point, distance) = query;
    let point: KdTreePoint = point.into();
    let all = container.within_radius(&point, distance.ceil() as i64);
    black_box(all.len())
}

fn kd_tree_query_bench_within_distance_exact(
    query: (Point, f64),
    count: usize,
    container: &KdTree2<KdTreePoint>,
) -> usize {
    let (point, distance) = query;
    let kd_point: KdTreePoint = point.into();
    let mut all: Vec<_> = container.within_radius(&kd_point, distance.ceil() as i64);
    all.select_nth_unstable_by_key(count - 1, |other| point.distance_2(&(**other).into()));
    black_box(all.len())
}

fn seg_tree_query_bench_within_distance(
    query: (Point, f64),
    container: &PointlikeContainer<Point>,
) -> usize {
    let (point, distance) = query;
    let mut count = 0;
    container.for_each_with_distance_from_point_at_most(&point, distance, |p| {
        black_box(p);
        count += 1;
    });
    count
}

fn seg_tree_query_bench_within_distance_exact(
    query: (Point, f64),
    count: usize,
    container: &PointlikeContainer<Point>,
) -> usize {
    let (point, distance) = query;
    let mut all = container.collect_with_distance_from_point_at_most(&point, distance);
    all.select_nth_unstable_by_key(count - 1, |other| point.distance_2(other));
    black_box(all.len())
}

fn rstar_query_nearest_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("rstar_nearest_query");

    let mut rng = XorShiftRng::from_seed(RNG_SEED);
    let containers = get_rtrees(10, NUM_POINTS, &mut rng);

    for &size in &[10, 30, 100, 300, 1000, 3000] {
        group.bench_function(size.to_string(), |b| {
            b.iter_with_setup(
                || {
                    let index = rng.gen_range(0..containers.len());
                    (get_random_query_with_count(size, &mut rng), index)
                },
                |(query, idx)| assert!(rstar_query_bench_nearest(query, &containers[idx]) >= size),
            );
        });
    }
    group.finish()
}

fn rstar_query_distance_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("rstar_distance_query");

    let mut rng = XorShiftRng::from_seed(RNG_SEED);
    let containers_lookup = get_rtrees(10, NUM_POINTS, &mut rng.clone());
    let containers = get_rtrees(10, NUM_POINTS, &mut rng);

    for &size in &[10, 30, 100, 300, 1000, 3000] {
        group.bench_function(size.to_string(), |b| {
            b.iter_with_setup(
                || {
                    let index = rng.gen_range(0..containers.len());
                    (
                        get_random_query_with_distance_2(size, &containers_lookup[index], &mut rng),
                        index,
                    )
                },
                |(query, index)| {
                    assert!(rstar_query_bench_within_distance(query, &containers[index]) >= size)
                },
            );
        });
    }
    group.finish()
}

fn rstar_query_distance_exact_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("rstar_distance_query_exact");

    let mut rng = XorShiftRng::from_seed(RNG_SEED);
    let containers_lookup = get_rtrees(10, NUM_POINTS, &mut rng.clone());
    let containers = get_rtrees(10, NUM_POINTS, &mut rng);

    for &size in &[10, 30, 100, 300, 1000, 3000] {
        group.bench_function(size.to_string(), |b| {
            b.iter_with_setup(
                || {
                    let index = rng.gen_range(0..containers.len());
                    (
                        get_random_query_with_distance_2(size, &containers_lookup[index], &mut rng),
                        size,
                        index,
                    )
                },
                |(query, size, index)| {
                    assert!(
                        rstar_query_bench_within_distance_exact(query, size, &containers[index])
                            >= size
                    )
                },
            );
        });
    }
    group.finish()
}

fn kd_tree_query_nearest_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("kd_tree_nearest_query");

    let mut rng = XorShiftRng::from_seed(RNG_SEED);
    let containers = get_kd_trees(10, NUM_POINTS, &mut rng);

    for &size in &[10, 30, 100, 300, 1000, 3000] {
        group.bench_function(size.to_string(), |b| {
            b.iter_with_setup(
                || {
                    let index = rng.gen_range(0..containers.len());
                    (get_random_query_with_count(size, &mut rng), index)
                },
                |(query, idx)| {
                    assert!(kd_tree_query_bench_nearest(query, &containers[idx]) >= size)
                },
            );
        });
    }
    group.finish()
}

fn kd_tree_query_distance_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("kd_tree_distance_query");

    let mut rng = XorShiftRng::from_seed(RNG_SEED);
    let containers_lookup = get_rtrees(10, NUM_POINTS, &mut rng.clone());
    let containers = get_kd_trees(10, NUM_POINTS, &mut rng);

    for &size in &[10, 30, 100, 300, 1000, 3000] {
        group.bench_function(size.to_string(), |b| {
            b.iter_with_setup(
                || {
                    let index = rng.gen_range(0..containers.len());
                    let (point, dist) =
                        get_random_query_with_distance_2(size, &containers_lookup[index], &mut rng);
                    ((point, (dist as f64).sqrt() + 1.0), index)
                },
                |(query, index)| {
                    assert!(kd_tree_query_bench_within_distance(query, &containers[index]) >= size)
                },
            );
        });
    }
    group.finish()
}

fn kd_tree_query_distance_exact_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("kd_tree_distance_query_exact");

    let mut rng = XorShiftRng::from_seed(RNG_SEED);
    let containers_lookup = get_rtrees(10, NUM_POINTS, &mut rng.clone());
    let containers = get_kd_trees(10, NUM_POINTS, &mut rng);

    for &size in &[10, 30, 100, 300, 1000, 3000] {
        group.bench_function(size.to_string(), |b| {
            b.iter_with_setup(
                || {
                    let index = rng.gen_range(0..containers.len());
                    let (point, dist) =
                        get_random_query_with_distance_2(size, &containers_lookup[index], &mut rng);
                    ((point, (dist as f64).sqrt() + 1.0), size, index)
                },
                |(query, size, index)| {
                    assert!(
                        kd_tree_query_bench_within_distance_exact(query, size, &containers[index])
                            >= size
                    )
                },
            );
        });
    }
    group.finish()
}

fn seg_tree_query_distance_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("seg_tree_distance_query");

    let mut rng = XorShiftRng::from_seed(RNG_SEED);
    let containers_lookup = get_rtrees(10, NUM_POINTS, &mut rng.clone());
    let containers = get_seg_trees(10, NUM_POINTS, &mut rng);

    for &size in &[10, 30, 100, 300, 1000, 3000] {
        group.bench_function(size.to_string(), |b| {
            b.iter_with_setup(
                || {
                    let index = rng.gen_range(0..containers.len());
                    let (point, dist) =
                        get_random_query_with_distance_2(size, &containers_lookup[index], &mut rng);
                    ((point, (dist as f64).sqrt() + 1.0), index)
                },
                |(query, index)| {
                    assert!(seg_tree_query_bench_within_distance(query, &containers[index]) >= size)
                },
            );
        });
    }
    group.finish()
}

fn seg_tree_query_distance_exact_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("seg_tree_distance_query_exact");

    let mut rng = XorShiftRng::from_seed(RNG_SEED);
    let containers_lookup = get_rtrees(10, NUM_POINTS, &mut rng.clone());
    let containers = get_seg_trees(10, NUM_POINTS, &mut rng);

    for &size in &[10, 30, 100, 300, 1000, 3000] {
        group.bench_function(size.to_string(), |b| {
            b.iter_with_setup(
                || {
                    let index = rng.gen_range(0..containers.len());
                    let (point, dist) =
                        get_random_query_with_distance_2(size, &containers_lookup[index], &mut rng);
                    ((point, (dist as f64).sqrt() + 1.0), size, index)
                },
                |(query, size, index)| {
                    assert!(
                        seg_tree_query_bench_within_distance_exact(query, size, &containers[index])
                            >= size
                    )
                },
            );
        });
    }
    group.finish()
}

criterion_group!(rstar_nearest, rstar_query_nearest_group);
criterion_group!(rstar_distance, rstar_query_distance_group);
criterion_group!(rstar_distance_exact, rstar_query_distance_exact_group);
criterion_group!(kd_tree_nearest, kd_tree_query_nearest_group);
criterion_group!(kd_tree_distance, kd_tree_query_distance_group);
criterion_group!(kd_tree_distance_exact, kd_tree_query_distance_exact_group);
criterion_group!(seg_tree_distance, seg_tree_query_distance_group);
criterion_group!(seg_tree_distance_exact, seg_tree_query_distance_exact_group);
criterion_main!(
    rstar_nearest,
    rstar_distance,
    rstar_distance_exact,
    kd_tree_nearest,
    kd_tree_distance,
    kd_tree_distance_exact,
    seg_tree_distance,
    seg_tree_distance_exact
);
