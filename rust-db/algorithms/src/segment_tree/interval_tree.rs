use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
    ops::Range,
};

use common::BpResult;

use super::{range::RangeExt, ImplicitSegmentTree, SegmentTree};

/// A custom interval tree implementation
/// Note that it has slightly different performance characteristics
/// than the standard version.
/// NB, each datum (type T) can only be associated with one interval.
#[derive(Clone, Debug)]
pub struct IntervalTree<T> {
    inner: ImplicitSegmentTree<HashSet<T>, fn() -> HashSet<T>>,
    data: HashMap<T, Range<u64>>,
}

impl<T: Eq + Clone + Hash + Debug> IntervalTree<T> {
    /// Create an empty interval tree with a given left and right border.
    /// #Runtime
    /// O(1)
    /// #Panics
    /// Panics if border_right <= border_left.
    pub fn new(range: Range<u64>) -> Self {
        println!("{:?}", range);
        Self {
            inner: ImplicitSegmentTree::with_default(range),
            data: HashMap::new(),
        }
    }

    /// Add a given interval with a datum.
    /// # Runtime
    /// O(log D) where D = border_right - border_left
    /// # Panics
    /// Panics if either target_left <= target_right does not hold or the bounds of the tree are exceeded.
    /// # Returns
    /// Returns Ok(()) if additon was successful.
    /// Returns an error if the datum was added previously.
    /// # Examples
    /// ```
    /// # use algorithms::segment_tree::IntervalTree;
    /// let mut tree = IntervalTree::new(0..9);
    /// assert!(tree.add(0..9, 0).is_ok());
    /// assert!(tree.add(3..4, 1).is_ok());
    /// assert!(tree.add(3..4, 2).is_ok());
    ///
    /// // 0 was added before
    /// assert!(tree.add(3..4, 0).is_err());
    /// ```
    /// ```rust,should_panic
    /// # use algorithms::segment_tree::IntervalTree;
    /// let mut tree = IntervalTree::new(0..9);
    /// tree.add(0..10, 0);
    /// ```
    pub fn add(&mut self, range: Range<u64>, datum: T) -> BpResult<()> {
        match self.data.get(&datum) {
            None => {
                self.data.insert(datum.clone(), range.clone());
                self.inner.range_query_mut(range, |tree, idx| {
                    tree[idx].insert(datum.clone());
                });
                Ok(())
            }
            Some(interval) => Err(format!(
                "IntervalTree: Trying to add datum {:?} that already exists at {:?}",
                datum, interval
            )
            .into()),
        }
    }

    /// Behaves exactly like [add](IntervalTree::add), but if the target_left is to small, we take border_left and if
    /// target_right is to big we take border_right.
    /// # Example
    /// ```
    /// # use algorithms::segment_tree::IntervalTree;
    /// let mut tree = IntervalTree::new(0..9);
    /// assert!(tree.add_clamped(0..10, 0).is_ok());
    /// ```
    pub fn add_clamped(&mut self, mut range: Range<u64>, datum: T) -> BpResult<()> {
        range.clamp_by(&self.inner.borders());

        self.add(range, datum)
    }

    /// Remove a datum from its associated interval.
    /// Note that this does not "shrink" the tree, but leaves behind empty nodes.
    /// # Runtime
    /// O(log D) where D = border_right - border_left
    /// # Returns
    /// Returns Ok(()) if deletion was successful.
    /// Returns an error if the datum was *not* added previously.
    /// # Examples
    /// ```
    /// # use algorithms::segment_tree::IntervalTree;
    /// let mut tree = IntervalTree::new(0..9);
    /// assert!(tree.add(0..9, 0).is_ok());
    /// assert_eq!(tree.query(5).count(), 1);
    ///
    /// assert!(tree.delete(&0).is_ok());
    /// assert_eq!(tree.query(5).count(), 0);
    ///
    /// assert!(tree.delete(&0).is_err());
    /// ```
    pub fn delete(&mut self, datum: &T) -> BpResult<()> {
        match self.data.get(datum) {
            Some(range) => {
                self.inner.range_query_mut(range.clone(), |tree, idx| {
                    tree[idx].remove(datum);
                });
                self.data.remove(datum);
                Ok(())
            }
            None => Err("IntervalTree: Trying to delete datum that does not exist in tree.".into()),
        }
    }

    /// Get the data associated with any interval that contains the point.
    /// # Runtime
    /// O(log D + k) where D = border_right - border_left and k is the output size
    /// Note that when intervals get large, so does k.
    pub fn query(&self, target: u64) -> impl Iterator<Item = &T> {
        assert!(self.inner.borders().contains(&target));
        let mut all_overlapping = Vec::new();
        self.inner.point_query(target, |_tree, idx| {
            all_overlapping.extend(self.inner[idx].iter())
        });
        all_overlapping.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::cmp::{max, min};

    prop_compose! {
        fn ordered_distinct_tuple(from: u64, to: u64)(
            x in from..to,
            y in from..to
        ) -> (u64, u64) {
            if x == y {
                return (from, to);
            }

            (min(x, y), max(x, y))
        }
    }

    prop_compose! {
        fn intervals()(
            intervals in prop::collection::vec(
                (ordered_distinct_tuple(0, u64::MAX - 1), any::<bool>()), 1..256)
        ) -> Vec<(Range<u64>, usize, bool)> {
            intervals
                .iter()
                .enumerate()
                .map(|(idx, ((x, y), to_delete))| {
                    (*x..*y, idx, *to_delete)
                })
                .collect()
        }
    }

    proptest! {
        #[test]
        fn test_random_intervals(
            intervals in intervals(),
            test_points in prop::collection::vec(0..u64::MAX-1, 1..256)
        ) {
            let mut tree = IntervalTree::<usize>::new(0.. u64::MAX);

            intervals.iter().for_each(|(range, datum, _)| {
                tree.add(range.clone(), *datum)
                    .expect("We're only adding valid intervals");
            });

            intervals.iter().for_each(|(_, datum, to_delete)| {
                if *to_delete {
                    tree.delete(datum)
                        .expect("We're only performing valid deletions");
                }
            });

            for point in test_points.iter() {
                let added: HashSet<usize> = intervals
                    .iter()
                    .filter(|(range, _, _)| range.start <= *point && range.end > *point)
                    .map(|(_, datum, _)| *datum)
                    .collect();

                let removed: HashSet<usize> = intervals
                    .iter()
                    .filter(|(_, _, to_delete)| *to_delete)
                    .filter(|(range, _, _)| range.start <= *point && range.end > *point)
                    .map(|(_, datum, _)| *datum)
                    .collect();

                let should_contain: HashSet<_> = added.difference(&removed).cloned().collect();
                let does_contain: HashSet<_> = tree.query(*point).cloned().collect();

                prop_assert_eq!(does_contain, should_contain);
            }
            prop_assert!(true);
        }
    }

    #[test]
    fn quick_test() -> BpResult<()> {
        let mut tree: IntervalTree<isize> = IntervalTree::new(0..u64::MAX);
        tree.add(0..1, 0)?;
        tree.add(0..1, 1)?;
        Ok(())
    }

    fn get_example_tree() -> IntervalTree<isize> {
        let mut tree = IntervalTree::<isize>::new(1..100);

        tree.add(1..10, -1).unwrap();
        tree.add(2..12, -2).unwrap();
        tree.add(12..17, -3).unwrap();

        tree
    }

    #[test]
    fn test_query() {
        let tree = get_example_tree();
        assert_eq!(tree.query(12).copied().collect::<Vec<_>>(), vec![-3]);
        assert_eq!(tree.query(17).count(), 0);
        assert_eq!(tree.query(42).count(), 0);
    }

    #[test]
    fn new_tree_is_empty() {
        let tree = IntervalTree::<usize>::new(0..2);

        assert!(tree.query(0).next() == None);
        assert!(tree.query(1).next() == None);
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn invalid_new_panics() {
        let left_greater_right = std::panic::catch_unwind(|| IntervalTree::<usize>::new(1..0));
        let left_is_right = std::panic::catch_unwind(|| IntervalTree::<usize>::new(0..0));
        assert!(left_greater_right.is_err());
        assert!(left_is_right.is_err());
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn invalid_add_panics() {
        let left_greater_right = std::panic::catch_unwind(|| get_example_tree().add(1..0, -42));
        let left_less_than_border = std::panic::catch_unwind(|| get_example_tree().add(0..10, -42));
        let right_greater_than_border =
            std::panic::catch_unwind(|| get_example_tree().add(1..101, -42));
        assert!(left_greater_right.is_err());
        assert!(left_less_than_border.is_err());
        assert!(right_greater_than_border.is_err());
    }

    #[test]
    fn invalid_query_panics() {
        let over_border = std::panic::catch_unwind(|| get_example_tree().query(420).count());
        let under_border = std::panic::catch_unwind(|| get_example_tree().query(0).count());
        let at_border = std::panic::catch_unwind(|| get_example_tree().query(100).count());

        assert!(over_border.is_err());
        assert!(under_border.is_err());
        assert!(at_border.is_err());
    }

    #[test]
    fn readd_datum() -> BpResult<()> {
        let mut tree = get_example_tree();
        tree.add(99..100, -42)?;
        assert!(tree.add(10..20, -42).is_err());
        tree.delete(&-42)?;
        tree.add(10..20, -42)?;
        Ok(())
    }

    #[test]
    fn invalid_delete_gives_error() {
        let mut tree = get_example_tree();
        assert!(tree.delete(&2).is_err());
    }
}
