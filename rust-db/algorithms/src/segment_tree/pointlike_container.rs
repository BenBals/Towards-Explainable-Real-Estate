use std::ops::Range;

use common::{BpError, BpResult, Pointlike};

use super::{range::RangeExt, ExplicitSegmentTree, SegmentTree};

/// A Container for [Pointlike]s which allows querying them by their location.
/// Their location have to be in a range which is known in advance.
/// This datastructure needs `O(xlen + n*log(xlen)*log(ylen))` space where `n` is the
/// number of contained points, `xlen` and `ylen` are the length of the respective range.
/// Therefore do not make x_range too huge.
#[derive(Debug, Clone)]
pub struct PointlikeContainer<P> {
    x_range: Range<u64>,
    y_range: Range<u64>,
    inner: ExplicitSegmentTree<Vec<P>>,
}

impl<P: Pointlike + Copy> PointlikeContainer<P> {
    /// Creates a new PointlikeContainer, which will accept Points in the specified range
    pub fn new(x_range: Range<u64>, y_range: Range<u64>) -> Self {
        Self {
            x_range: x_range.clone(),
            y_range,
            inner: ExplicitSegmentTree::with_default(x_range),
        }
    }

    /// Creates a new PointlikeContainer, with all the given Points inserted.
    /// The acceptable ranges will be minimal to contain the given points.
    pub fn with<IntoIter, Iter>(points: IntoIter) -> BpResult<Self>
    where
        IntoIter: IntoIterator<IntoIter = Iter, Item = P>,
        Iter: Clone + Iterator<Item = P>,
    {
        let iter = points.into_iter();
        let x_vals = iter.clone().map(|p| p.x());
        let y_vals = iter.clone().map(|p| p.y());

        let x_range = x_vals
            .clone()
            .min()
            .ok_or_else::<BpError, _>(|| "No points".into())?
            ..(x_vals
                .max()
                .ok_or_else::<BpError, _>(|| "No points".into())?
                + 1);
        let y_range = y_vals
            .clone()
            .min()
            .ok_or_else::<BpError, _>(|| "No points".into())?
            ..(y_vals
                .max()
                .ok_or_else::<BpError, _>(|| "No points".into())?
                + 1);

        let mut res = Self::new(x_range, y_range);
        for point in iter {
            res.insert(point).unwrap();
        }
        res.finalize();
        Ok(res)
    }
}

impl<P> PointlikeContainer<P>
where
    P: Pointlike + Copy,
{
    /// Inserts the given Point
    /// # Runtime
    /// `O(log(xlen)log(ylen))`
    /// # Return
    /// This function returns an Error if the point lies out of the covered range.
    /// # Invariants
    /// Calling this invalidates the datastructure as the invariant that all Segmentree nodes are sorted vectors is not upheld.
    /// This instances becomes valid again once [finalize] has been called.
    fn insert(&mut self, point: P) -> BpResult<()> {
        if !self.x_range.contains(&point.x()) || !self.y_range.contains(&point.y()) {
            return Err(format!(
                "Point({}, {}) has coordinates outside of the range of this container({:?}, {:?})",
                point.x(),
                point.y(),
                self.x_range,
                self.y_range
            )
            .into());
        }
        self.inner.point_query_mut(point.x(), |x_tree, x_idx| {
            x_tree[x_idx].push(point);
        });
        Ok(())
    }

    /// If this instance got invalidated this function makes it valid again.
    fn finalize(&mut self) {
        self.inner
            .data
            .iter_mut()
            .for_each(|vec| vec.sort_by_key(|i| i.y()));
    }

    /// Will call `f` for every Point which is contained in both respective ranges.
    /// # Runtime
    /// `O(log(xlen)log(ylen) + k)` where `k` is the number of function calls.
    pub fn for_each_in_range<F>(&self, mut x_query: Range<u64>, mut y_query: Range<u64>, mut f: F)
    where
        F: FnMut(P),
    {
        x_query.clamp_by(&self.x_range);
        y_query.clamp_by(&self.y_range);

        self.inner.range_query(x_query, |x_tree, x_idx| {
            let y_sorted_elements = &x_tree[x_idx];

            let mid_position: usize =
                match y_sorted_elements.binary_search_by_key(&y_query.start, |i| i.y()) {
                    Ok(pos) => pos as usize,
                    Err(pos) => pos as usize,
                };
            for pos in (0..mid_position)
                .rev()
                .take_while(|&pos| y_query.contains(&y_sorted_elements[pos].y()))
                .chain(
                    (mid_position..y_sorted_elements.len())
                        .take_while(|&pos| y_query.contains(&y_sorted_elements[pos].y())),
                )
            {
                f(y_sorted_elements[pos]);
            }
        })
    }

    /// Will call `f` for every Point with euclidean distance at most `distance` from
    /// the center point.
    /// # Runtime
    /// `O(log(xlen) log(ylen) + k)` where `k` is the number of Points contained in the
    /// smallest square enclosing a circle with radius of `distance` around the center point.
    pub fn for_each_with_distance_at_most<F>(
        &self,
        center_x: u64,
        center_y: u64,
        distance: f64,
        mut f: F,
    ) where
        F: FnMut(P),
    {
        let dist_u64 = distance.ceil() as u64;
        let query_x_start = center_x.saturating_sub(dist_u64);
        let query_x_end = center_x.saturating_add(dist_u64.saturating_add(1));

        let query_y_start = center_y.saturating_sub(dist_u64);
        let query_y_end = center_y.saturating_add(dist_u64.saturating_add(1));

        // This should not be ceiled as we want to test that the squared distance is at most distance**2.
        // As the squared distance always is an integer the default behaviour of rounding to zero is exactly what we want.
        // Floating point errors are _not_ considered here.
        let dist_squared_i64 = (distance * distance) as i64;

        self.for_each_in_range(
            query_x_start..query_x_end,
            query_y_start..query_y_end,
            |p| {
                let dx = p.x() as i64 - center_x as i64;
                let dy = p.y() as i64 - center_y as i64;
                if dx * dx + dy * dy <= dist_squared_i64 {
                    f(p)
                }
            },
        );
    }

    /// Like [for_each_with_distance_at_most], but allows to specify the center point as a Point.
    pub fn for_each_with_distance_from_point_at_most<F>(&self, point: &P, distance: f64, f: F)
    where
        F: FnMut(P),
    {
        self.for_each_with_distance_at_most(point.x(), point.y(), distance, f);
    }

    /// Like [for_each_in_range], but returns a Vec of all accessed points.
    pub fn collect_in_range(&self, x_query: Range<u64>, y_query: Range<u64>) -> Vec<P> {
        let mut res = Vec::new();
        self.for_each_in_range(x_query, y_query, |p| res.push(p));
        res
    }

    /// Like [for_each_with_distance_at_most], but returns a Vec of all accessed points.
    pub fn collect_with_distance_at_most(
        &self,
        center_x: u64,
        center_y: u64,
        distance: f64,
    ) -> Vec<P> {
        let mut res = Vec::new();
        self.for_each_with_distance_at_most(center_x, center_y, distance, |p| res.push(p));
        res
    }

    /// Like [for_each_with_distance_from_point_at_most], but returns a Vec of all accessed points.
    pub fn collect_with_distance_from_point_at_most(&self, point: &P, distance: f64) -> Vec<P> {
        let mut res = Vec::new();
        self.for_each_with_distance_from_point_at_most(point, distance, |p| res.push(p));
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::Pointlike;
    use proptest::prelude::*;
    use std::ops::Range;
    use test_helpers::Point;

    prop_compose! {
        fn sub_range(range: Range<u64>)(a in range.clone(),
                                        b in range,)
            -> Range<u64> {
            a.min(b)..a.max(b)
        }
    }

    prop_compose! {
        fn point_in_range(range: Range<u64>)(x in range.clone(), y in range, key in prop::num::usize::ANY) -> Point {
            Point::new(x, y, key)
        }
    }

    #[test]
    fn does_not_panic_on_invalid_range() {
        let container = PointlikeContainer::<Point>::new(5..6, 5..6);

        container.for_each_in_range(0..9, 0..9, |_| {});
        container.for_each_in_range(0..1, 0..9, |_| {});
        container.for_each_in_range(7..9, 0..9, |_| {});

        container.for_each_in_range(0..9, 0..1, |_| {});
        container.for_each_in_range(0..1, 0..1, |_| {});
        container.for_each_in_range(7..9, 0..1, |_| {});

        container.for_each_in_range(0..9, 7..9, |_| {});
        container.for_each_in_range(0..1, 7..9, |_| {});
        container.for_each_in_range(7..9, 7..9, |_| {});
    }

    proptest! {
        #![proptest_config(ProptestConfig {
        cases: 100000, .. ProptestConfig::default()
        })]
        #[test]
        fn collect_in_range_gives_correct(
            points in prop::collection::vec(point_in_range(500..1500), 0..512),
            x_range in sub_range(0..2000),
            y_range in sub_range(0..2000),
        ) {
            let container = PointlikeContainer::with(points.iter().copied()).unwrap_or_else(|_| PointlikeContainer::new(0..1, 0..1));

            let queried = container.collect_in_range(x_range.clone(), y_range.clone());

            for point in &points {
                if x_range.contains(&point.x()) && y_range.contains(&point.y()) {
                    prop_assert!(queried.contains(point));
                } else {
                    prop_assert!(!queried.contains(point));
                }
            }
        }

        #[test]
        fn collect_with_distance_from_point_at_most_gives_correct(
            points in prop::collection::vec(point_in_range(500..1500), 0..512),
            center in point_in_range(0..2000),
            radius in 0..2000
        ) {
            let container = PointlikeContainer::with(points.iter().copied()).unwrap_or_else(|_| PointlikeContainer::new(0..1, 0..1));

            let queried = container.collect_with_distance_from_point_at_most(&center, radius as f64);

            for point in &points {
                let dx = center.x() as f64 - point.x() as f64;
                let dy = center.y() as f64 - point.y() as f64;
                if dx * dx + dy *dy <= (radius * radius) as f64 {
                    prop_assert!(queried.contains(point));
                } else {
                    prop_assert!(!queried.contains(point));
                }
            }
        }

        #[test]
        fn collect_with_distance_at_most_point_gives_correct(
            points in prop::collection::vec(point_in_range(500..1500), 0..512),
            center in point_in_range(0..2000),
            radius in 0..2000
        ) {
            let container = PointlikeContainer::with(points.iter().copied()).unwrap_or_else(|_| PointlikeContainer::new(0..1, 0..1));

            let queried = container.collect_with_distance_at_most(center.x(), center.y(), radius as f64);

            for point in &points {
                let dx = center.x() as f64 - point.x() as f64;
                let dy = center.y() as f64 - point.y() as f64;
                if dx * dx + dy *dy <= (radius * radius) as f64 {
                    prop_assert!(queried.contains(point));
                } else {
                    prop_assert!(!queried.contains(point));
                }
            }
        }
    }
}
