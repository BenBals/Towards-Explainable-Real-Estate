use std::ops::{Deref, Index, IndexMut, Range};

use derive_more::*;
use typed_index_collections::TiVec;

use super::{RangeExt, SegmentTree};

// An Index into a ExplicitSegmentTree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, From, Into)]
pub struct ExplicitSegmentTreeNodeIdx(usize);

/// A SegmentTree which initially creates all its nodes.
#[derive(Debug, Clone)]
pub struct ExplicitSegmentTree<T> {
    left: u64,
    len: usize,
    pub(crate) data: TiVec<ExplicitSegmentTreeNodeIdx, T>,
}

impl<T> ExplicitSegmentTree<T> {
    /// Creates a new ExplicitSegmentTree.
    /// For every node the function `f` will get called once to produce its initial value.
    /// # Panics
    /// Let `b` be the smallest power of two at least as big as the range.
    /// Then the this function panics if `log2(b) > 31` or if `range.start + b > u64::MAX`
    /// or if the supplied range is empty.
    /// # Remarks
    /// The range covered by this segement tree will be `range.start..range.start + b`.
    pub fn with_generator(range: Range<u64>, mut f: impl FnMut() -> T) -> Self {
        assert!(!range.is_empty());

        // 32 stands for usize::BITS, but this is currently unstable
        // changing this would allow *really* huge ExplicitSegmentTrees which is propably not
        // what one wants
        //
        // finds the smallest power of two atleast as big as right - left
        let len = (0..32 - 1)
            .map(|b| 1usize << b)
            .find(|len| range.end - range.start <= *len as u64)
            .unwrap();
        assert!(range.start as u128 + len as u128 <= u64::MAX as u128);
        Self {
            left: range.start,
            len,
            data: (0..2 * len).map(|_| f()).collect(),
        }
    }

    /// Creates a new ExplicitSegmenTree, where every Node initially has the value `value`.
    /// # Panics & Remarks
    /// Refer to [with_generator].
    pub fn with_value(range: Range<u64>, value: T) -> Self
    where
        T: Clone,
    {
        Self::with_generator(range, || value.clone())
    }

    /// Creates a new ExplicitSegmenTree, where every Node is initialized with the default of this type.
    /// # Panics & Remarks
    /// Refer to [with_generator].
    pub fn with_default(range: Range<u64>) -> Self
    where
        T: Default,
    {
        Self::with_generator(range, T::default)
    }

    // The weird type of the function is needed to abstract over &self and &mut self.
    // The return type of f has to be B to retain ownership as &mut self does *not* implement Copy.
    fn point_query_inner<D: Deref<Target = Self>>(
        mut zelf: D,
        position: u64,
        mut f: impl FnMut(D, ExplicitSegmentTreeNodeIdx) -> D,
    ) {
        assert!(zelf.borders().contains(&position));
        // The downcast is valid since len is bounded by 1 << 30 and we don't care about 16 bit targets.
        let start_idx = (position - zelf.left + zelf.len as u64) as usize;

        for pos in (0..64).map(|s| start_idx >> s).take_while(|pos| *pos > 0) {
            zelf = f(zelf, ExplicitSegmentTreeNodeIdx(pos));
        }
    }

    fn range_query_inner<D: Deref<Target = Self>>(
        mut zelf: D,
        query_range: &Range<u64>,
        mut f: impl FnMut(D, ExplicitSegmentTreeNodeIdx) -> D,
    ) {
        let root_range = zelf.borders();
        assert!(query_range.is_normal());
        if query_range.is_empty() {
            return;
        }
        assert!(root_range.is_superset(query_range));
        // The downcast is valid since len is bounded by 1 << 30 and we don't care about 16 bit targets.
        let mut left_idx = (query_range.start - zelf.left + zelf.len as u64) as usize;
        let mut right_idx = (query_range.end - zelf.left + zelf.len as u64) as usize;

        while left_idx < right_idx {
            if left_idx % 2 == 1 {
                zelf = f(zelf, ExplicitSegmentTreeNodeIdx(left_idx));
                left_idx += 1;
            }

            if right_idx % 2 == 1 {
                right_idx -= 1;
                zelf = f(zelf, ExplicitSegmentTreeNodeIdx(right_idx));
            }

            left_idx >>= 1;
            right_idx >>= 1;
        }
    }
}

impl<T> Index<ExplicitSegmentTreeNodeIdx> for ExplicitSegmentTree<T> {
    type Output = T;

    fn index(&self, index: ExplicitSegmentTreeNodeIdx) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<ExplicitSegmentTreeNodeIdx> for ExplicitSegmentTree<T> {
    fn index_mut(&mut self, index: ExplicitSegmentTreeNodeIdx) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T> SegmentTree<T> for ExplicitSegmentTree<T> {
    type NodeIdx = ExplicitSegmentTreeNodeIdx;

    fn borders(&self) -> Range<u64> {
        self.left..(self.left + self.len as u64)
    }

    fn point_query(&self, position: u64, mut f: impl FnMut(&Self, Self::NodeIdx)) {
        Self::point_query_inner(self, position, |zelf, idx| {
            f(zelf, idx);
            zelf
        });
    }

    fn point_query_mut(&mut self, position: u64, mut f: impl FnMut(&mut Self, Self::NodeIdx)) {
        Self::point_query_inner(self, position, |zelf, idx| {
            f(zelf, idx);
            zelf
        });
    }

    fn range_query(&self, range: Range<u64>, mut f: impl FnMut(&Self, Self::NodeIdx)) {
        Self::range_query_inner(self, &range, |zelf, idx| {
            f(zelf, idx);
            zelf
        })
    }

    fn range_query_mut(&mut self, range: Range<u64>, mut f: impl FnMut(&mut Self, Self::NodeIdx)) {
        Self::range_query_inner(self, &range, |zelf, idx| {
            f(zelf, idx);
            zelf
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ExplicitSegmentTree;

    #[test]
    #[should_panic]
    fn huge_range_panics() {
        ExplicitSegmentTree::<usize>::with_default(0..1 << 31);
    }

    #[test]
    fn huge_value_with_small_range_works() {
        ExplicitSegmentTree::<usize>::with_default(u64::MAX - 1..u64::MAX);
    }
}
