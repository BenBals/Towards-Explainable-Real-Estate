//! This module contains the sweepline algorithm based on IntervalTree

use std::{
    cmp::{max, min, Ordering},
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    marker::PhantomData,
};

use common::{Keyed, Pointlike};

use super::segment_tree::IntervalTree;

#[derive(Debug)]
enum Event<'i, Point, Key>
where
    Point: Pointlike + Keyed<Key = Key>,
    Key: Clone,
{
    Enter(&'i Point, PhantomData<Key>),
    Query(&'i Point, PhantomData<Key>),
    Leave(&'i Point, PhantomData<Key>),
}

// returns a + b clamped to 0 and u64::MAX
fn add_clamped(a: u64, b: u64) -> u64 {
    min(a, u64::MAX - b) + b
}

// returns a - b clamped to 0 and u64::MAX
fn subtract_clamped(a: u64, b: u64) -> u64 {
    max(a, b) - b
}

impl<'i, Point, Key> Event<'i, Point, Key>
where
    Point: Pointlike + Keyed<Key = Key>,
    Key: Clone,
{
    fn x_with_radius(&self, radius: u64) -> u64 {
        match self {
            Event::Enter(p, _) => subtract_clamped(p.x(), radius),
            Event::Query(p, _) => p.x(),
            Event::Leave(p, _) => add_clamped(p.x(), radius),
        }
    }

    fn variant_comparison_value(&self) -> u64 {
        match self {
            Event::Enter(_, _) => 0,
            Event::Query(_, _) => 1,
            Event::Leave(_, _) => 2,
        }
    }

    fn compare(&self, other: &Self, radius: u64) -> Ordering {
        (self.x_with_radius(radius), self.variant_comparison_value()).cmp(&(
            other.x_with_radius(radius),
            other.variant_comparison_value(),
        ))
    }
}

fn abs_diff(a: u64, b: u64) -> u64 {
    max(a, b) - min(a, b)
}

fn is_distance_at_most<Key: Clone, Point: Pointlike + Keyed<Key = Key>>(
    p1: &Point,
    p2: &Point,
    radius: u64,
) -> bool {
    let diff_x: u128 = abs_diff(p1.x(), p2.x()).into();
    let diff_y: u128 = abs_diff(p1.y(), p2.y()).into();
    let radius: u128 = radius.into();

    if diff_y > radius {
        return false;
    }
    diff_x * diff_x <= radius * radius - diff_y * diff_y
}

/// Performs an action for every point and all its neighbors inside a given radius
///
/// # Arguments
/// * points: a vector of Pointlikes which should be considered as center points and neighbors
/// * radius: the radius size which determines whether two points are neighbors
/// * action: a closure which receives the key of a point and the keys of all its neighbors including the point itself
///
/// Runtime: O(n log(D) + k) where n is the number of points,
/// D is (maximum y coordinate - minimum y coordinate) and k is the sum of the sizes of the neighbors for every point
///
/// # Examples
/// ```
/// use test_helpers::Point;
/// use algorithms::sweepline::for_every_close_point_do;
/// let points = vec![Point::new(0,0,0) , Point::new(1,2,1), Point::new(3,3,2)];
///
/// for_every_close_point_do(&points, 3, |key, neighbors| {
///     match key {
///         0 => assert_eq!(neighbors, vec![1]),
///         1 => assert!(neighbors.contains(&0) && neighbors.contains(&2) && !neighbors.contains(&1)),
///         2 => assert_eq!(neighbors, vec![1]),
///         _ => unreachable!(),
///     }
/// });
/// ```
///
/// Note that if the radius is too big, k is quadratic in n: You can approximate k by n * radius^2/D^2.
///
/// Note that none of the points can have u64::MAX as y coordinate, since that won't fit inside
/// an interval tree
pub fn for_every_close_point_do<Point, Key, F>(points: &[Point], radius: u64, mut action: F)
where
    Point: Pointlike + Keyed<Key = Key>,
    F: FnMut(Key, Vec<Key>),
    Key: Clone + Hash + Eq + Debug,
{
    if points.is_empty() {
        return;
    }

    let mut key_point_map = HashMap::new();
    for point in points {
        key_point_map.insert(point.key(), point);
    }

    let mut events: Vec<_> = points
        .iter()
        .flat_map(|point| {
            vec![
                Event::Enter(point, PhantomData::default()),
                Event::Query(point, PhantomData::default()),
                Event::Leave(point, PhantomData::default()),
            ]
        })
        .collect();

    events.sort_unstable_by(|e1, e2| e1.compare(e2, radius));

    let mut tree: IntervalTree<Key> = IntervalTree::new(
        points
            .iter()
            .map(|p| p.y())
            .min()
            .expect("Points are empty")
            ..(points
                .iter()
                .map(|p| p.y())
                .max()
                .expect("Points are empty")
                + 1),
    );

    for event in events {
        match event {
            Event::Enter(point, _) => tree
                .add_clamped(
                    // We add one since the intervals in the tree are right-exclusive.
                    subtract_clamped(point.y(), radius)
                        ..add_clamped(add_clamped(point.y(), radius), 1),
                    point.key().clone(),
                )
                .expect("Adding point to IntervalTree failed"),
            Event::Query(point, _) => action(
                point.key().clone(),
                tree.query(point.y())
                    .cloned()
                    .filter(|key| *key != point.key())
                    .filter(|key| {
                        is_distance_at_most(
                            point,
                            key_point_map
                                .get(key)
                                .expect("We only get keys we inserted before"),
                            radius,
                        )
                    })
                    .collect(),
            ),
            Event::Leave(point, _) => tree
                .delete(&point.key())
                .expect("Deletion from IntervalTree failed"),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;
    use test_helpers::Point;

    prop_compose! {
        fn points()
            (coords in prop::collection::vec((0..u64::MAX, 0..(u64::MAX-1)), 0..256)
        ) -> Vec<Point> {
            coords
                .iter()
                .enumerate()
                .map(|(idx, (x, y))| Point::new(*x, *y, idx))
                .collect()
        }
    }

    proptest! {
        #[test]
        fn test_random_points(
            points in points(),
            radius in 1..u64::MAX
        ) {
            for_every_close_point_do(&points, radius, |key, keys| {
                let myself = &points[key];
                let matching_dully: HashSet<usize> = points
                    .iter()
                    .filter(|other| myself.key() != other.key())
                    .filter(|other| is_distance_at_most(myself, *other, radius))
                    .map(|other| other.key())
                    .collect();

                let matching_sweepline: HashSet<usize> = keys.iter().cloned().collect();
                assert_eq!(matching_dully, matching_sweepline);
            });
        }
    }
}
