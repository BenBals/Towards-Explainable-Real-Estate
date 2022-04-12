//! This module contains everything related to the Segment Tree datastructure.

use std::ops::{Index, IndexMut, Range};

/// A trait which abstract over the SegmentTree types.
/// To retrieve or change information in the segment tree the `*query*` functions are used.
/// They receive a closure which takes as argument a (mutable) reference to this tree and
/// an index of the current node.
/// # Remark
/// If you need to combine range update and range queries be sure this is what you want to do.
/// Usually you would need lazy propagation to accomplish this.
/// Currently this is not supported.
/// # Example
/// Let's build a static Interval Tree, which gets intervals and for a given position should give
/// the number of intervals containing this position.
/// This example uses ExplicitSegmentTree as a type implementing segment tree.
/// ```
/// # use std::ops::Range;
/// # use algorithms::segment_tree::*;
/// pub struct ExampleTree {
///     inner: ExplicitSegmentTree<u64>,
/// }
/// impl ExampleTree {
///     fn with(intervals: &[Range<u64>]) -> Self {
///         let start = intervals.iter().map(|r| r.start).min().expect("Need at least one interval");
///         let end = intervals.iter().map(|r| r.end).max().expect("Need at least one interval");
///
///         let mut inner = ExplicitSegmentTree::with_default(start..end);
///         for range in intervals {
///             inner.range_query_mut(range.clone(), |tree: &mut ExplicitSegmentTree<u64>, idx| {
///                 tree[idx] += 1u64
///             });
///         }
///
///         Self { inner }
///     }
///
///     fn num_overlapping(&self, position: u64) -> u64 {
///         let mut res = 0;
///         if self.inner.borders().contains(&position) {
///             self.inner.point_query(position, |tree, idx| res += tree[idx]);
///             res
///         } else {
///             0
///         }
///     }
/// }
/// let tree = ExampleTree::with(&[2..5, 3..7, 8..8]);
/// assert_eq!(tree.num_overlapping(0), 0);
/// assert_eq!(tree.num_overlapping(2), 1);
/// assert_eq!(tree.num_overlapping(4), 2);
/// assert_eq!(tree.num_overlapping(6), 1);
/// assert_eq!(tree.num_overlapping(8), 0);
/// ```
pub trait SegmentTree<T>:
    Index<<Self as SegmentTree<T>>::NodeIdx, Output = T>
    + IndexMut<<Self as SegmentTree<T>>::NodeIdx, Output = T>
{
    /// Nodes of the SegmentTree are referred to by this identifier.
    type NodeIdx: Copy;

    /// Gives the range for which the segment tree holds values.
    /// This might differ from the range supplied for creation,
    /// but must be a superset of this range.
    fn borders(&self) -> Range<u64>;

    /// This query calls `f` on all nodes which cover `position`.
    /// This should be done at most O(log(self.borders().len())) times.
    /// # Panics
    /// This function should panic if `self.borders(&self.contains)` does not holds.
    fn point_query(&self, position: u64, f: impl FnMut(&Self, Self::NodeIdx));
    /// Like [point_query], but provides mutable access.
    fn point_query_mut(&mut self, position: u64, f: impl FnMut(&mut Self, Self::NodeIdx));

    /// This query calls f on all nodes which are contained in the range, whose parent nodes are not.
    /// This should be done at most O(log(self.borders().len())) times.
    /// # Panics
    /// This function should panic if there are positions contained in `range`, which are not contained in `self.borders()`.
    /// This function can panic if `range` is empty, but not inside `self.borders()`.
    fn range_query(&self, range: Range<u64>, f: impl FnMut(&Self, Self::NodeIdx));
    /// Like [range_query], but provides mutable access.
    fn range_query_mut(&mut self, range: Range<u64>, f: impl FnMut(&mut Self, Self::NodeIdx));
}

mod explicit;
pub use explicit::ExplicitSegmentTree;

mod implicit;
pub use implicit::ImplicitSegmentTree;

mod interval_tree;
pub use interval_tree::IntervalTree;

mod pointlike_container;
pub use pointlike_container::PointlikeContainer;

mod range;
use range::RangeExt;

#[cfg(test)]
mod tests;
