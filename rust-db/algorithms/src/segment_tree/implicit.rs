use std::{
    num::NonZeroUsize,
    ops::{Deref, Index, IndexMut, Range},
};

use super::{RangeExt, SegmentTree};

/// An Index into an ImplicitSegmentTree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// Using NonZeroUsize decreases the raw memory of ImplicitSegmentTreeNode by half
pub struct ImplicitSegmentTreeNodeIdx(NonZeroUsize);

impl From<usize> for ImplicitSegmentTreeNodeIdx {
    fn from(u: usize) -> Self {
        Self(NonZeroUsize::new(u + 1).unwrap())
    }
}
impl From<ImplicitSegmentTreeNodeIdx> for usize {
    fn from(idx: ImplicitSegmentTreeNodeIdx) -> Self {
        Into::<usize>::into(idx.0) - 1
    }
}

impl ImplicitSegmentTreeNodeIdx {
    pub fn into_usize(self) -> usize {
        Into::<usize>::into(self)
    }
}

#[derive(Debug, Clone)]
struct ImplicitSegmentTreeNode<T> {
    left: Option<ImplicitSegmentTreeNodeIdx>,
    right: Option<ImplicitSegmentTreeNodeIdx>,
    inner: T,
}

impl<T> ImplicitSegmentTreeNode<T> {
    fn new(inner: T) -> Self {
        Self {
            left: None,
            right: None,
            inner,
        }
    }
}

/// A SegmentTree which creates its nodes lazily.
/// A Node will be created the first time it is needed in either
/// `point_query_mut` or `range_query_mut`.
#[derive(Debug, Clone)]
pub struct ImplicitSegmentTree<T, G> {
    range: Range<u64>,
    generator: G,
    data: Vec<ImplicitSegmentTreeNode<T>>,
}

impl<T, G: FnMut() -> T> ImplicitSegmentTree<T, G> {
    /// Creates a new ImplicitSegmentTree where each new node will
    /// initially be assigned a value produced by `generator`.
    /// # Panics
    /// If the supplied range is empty.
    pub fn with_generator(range: Range<u64>, mut generator: G) -> ImplicitSegmentTree<T, G> {
        assert!(!range.is_empty());
        let data = vec![ImplicitSegmentTreeNode::new(generator())];
        ImplicitSegmentTree {
            range,
            generator,
            data,
        }
    }
}

impl<T: Default> ImplicitSegmentTree<T, fn() -> T> {
    /// Creates a new ImplicitSegmentTree where each new node will
    /// initially be assigned the default value of this type.
    /// # Panics
    /// If the supplied range is empty.
    pub fn with_default(range: Range<u64>) -> Self {
        ImplicitSegmentTree::with_generator(range, Default::default)
    }
}

impl<T, G: Fn() -> T> ImplicitSegmentTree<T, G> {
    // This creates the left child of node `idx` if it does not exist and `idx` is no leaf.
    fn do_not_create_children(
        &self,
        _range: &Range<u64>,
        _idx: ImplicitSegmentTreeNodeIdx,
    ) -> &Self {
        self
    }

    // This creates the left child of node `idx` if it does not exist and `idx` is no leave.
    fn create_left_child(
        &mut self,
        range: &Range<u64>,
        idx: ImplicitSegmentTreeNodeIdx,
    ) -> &mut Self {
        if self.data[idx.into_usize()].left.is_none() && range.is_splittable() {
            let new = ImplicitSegmentTreeNode::new((self.generator)());
            self.data.push(new);
            let new_left = Some(self.data.len() - 1);
            self.data[Into::<usize>::into(idx)].left = new_left.map(Into::into);
        }
        self
    }

    // This creates the right child of node `idx` if it does not exist and `idx` is no leave.
    fn create_right_child(
        &mut self,
        range: &Range<u64>,
        idx: ImplicitSegmentTreeNodeIdx,
    ) -> &mut Self {
        if self.data[idx.into_usize()].right.is_none() && range.is_splittable() {
            let new = ImplicitSegmentTreeNode::new((self.generator)());
            self.data.push(new);
            let new_right = Some(self.data.len() - 1);
            self.data[Into::<usize>::into(idx)].right = new_right.map(Into::into);
        }
        self
    }

    // The weird type of the function is needed to abstract over &self and &mut self.
    // The return type of f has to be D to retain ownership as &mut self does *not* implement Copy.
    fn point_query_inner<D: Deref<Target = Self>>(
        mut zelf: D,
        position: u64,
        node_range: &Range<u64>,
        node_idx: ImplicitSegmentTreeNodeIdx,
        mut f: impl FnMut(D, ImplicitSegmentTreeNodeIdx) -> D,
        might_create_left_child: fn(D, &Range<u64>, ImplicitSegmentTreeNodeIdx) -> D,
        might_create_right_child: fn(D, &Range<u64>, ImplicitSegmentTreeNodeIdx) -> D,
    ) {
        // relies on the invariant that the queried position and the node range overlap.
        zelf = f(zelf, node_idx);
        if let Some((left_range, right_range)) = node_range.split() {
            if left_range.contains(&position) {
                zelf = might_create_left_child(zelf, node_range, node_idx);
                if let Some(left_child) = zelf.data[node_idx.into_usize()].left {
                    Self::point_query_inner(
                        zelf,
                        position,
                        &left_range,
                        left_child,
                        f,
                        might_create_left_child,
                        might_create_right_child,
                    )
                }
            } else {
                zelf = might_create_right_child(zelf, node_range, node_idx);
                if let Some(right_child) = zelf.data[node_idx.into_usize()].right {
                    Self::point_query_inner(
                        zelf,
                        position,
                        &right_range,
                        right_child,
                        f,
                        might_create_left_child,
                        might_create_right_child,
                    )
                }
            }
        }
    }

    #[allow(clippy::branches_sharing_code)]
    fn range_query_inner<D, F>(
        mut zelf: D,
        query_range: &Range<u64>,
        node_range: &Range<u64>,
        node_idx: ImplicitSegmentTreeNodeIdx,
        mut f: F,
        might_create_left_child: fn(D, &Range<u64>, ImplicitSegmentTreeNodeIdx) -> D,
        might_create_right_child: fn(D, &Range<u64>, ImplicitSegmentTreeNodeIdx) -> D,
    ) -> (D, F)
    where
        D: Deref<Target = Self>,
        F: FnMut(D, ImplicitSegmentTreeNodeIdx) -> D,
    {
        if query_range.is_superset(node_range) {
            // this is a minimal overlapping segment so stop recursion here.
            zelf = f(zelf, node_idx);
            (zelf, f)
        } else {
            if let Some((left_range, right_range)) = node_range.split() {
                if left_range.intersects(query_range) {
                    zelf = might_create_left_child(zelf, node_range, node_idx);
                    if let Some(left_child) = zelf.data[node_idx.into_usize()].left {
                        let (zelf_, f_) = Self::range_query_inner(
                            zelf,
                            query_range,
                            &left_range,
                            left_child,
                            f,
                            might_create_left_child,
                            might_create_right_child,
                        );
                        zelf = zelf_;
                        f = f_;
                    }
                }

                if right_range.intersects(query_range) {
                    zelf = might_create_right_child(zelf, node_range, node_idx);
                    if let Some(right_child) = zelf.data[node_idx.into_usize()].right {
                        let (zelf_, f_) = Self::range_query_inner(
                            zelf,
                            query_range,
                            &right_range,
                            right_child,
                            f,
                            might_create_left_child,
                            might_create_right_child,
                        );
                        zelf = zelf_;
                        f = f_;
                    }
                }
            }
            (zelf, f)
        }
    }
}

impl<T, G> Index<ImplicitSegmentTreeNodeIdx> for ImplicitSegmentTree<T, G> {
    type Output = T;

    fn index(&self, index: ImplicitSegmentTreeNodeIdx) -> &Self::Output {
        &self.data[index.into_usize()].inner
    }
}

impl<T, G> IndexMut<ImplicitSegmentTreeNodeIdx> for ImplicitSegmentTree<T, G> {
    fn index_mut(&mut self, index: ImplicitSegmentTreeNodeIdx) -> &mut Self::Output {
        &mut self.data[index.into_usize()].inner
    }
}

impl<T, G: Fn() -> T> SegmentTree<T> for ImplicitSegmentTree<T, G> {
    type NodeIdx = ImplicitSegmentTreeNodeIdx;

    fn borders(&self) -> Range<u64> {
        self.range.clone()
    }

    fn point_query(&self, position: u64, mut f: impl FnMut(&Self, Self::NodeIdx)) {
        assert!(self.range.contains(&position));
        Self::point_query_inner(
            self,
            position,
            &self.range,
            0.into(),
            |zelf, idx| {
                f(zelf, idx);
                zelf
            },
            Self::do_not_create_children,
            Self::do_not_create_children,
        );
    }

    fn point_query_mut(&mut self, position: u64, mut f: impl FnMut(&mut Self, Self::NodeIdx)) {
        assert!(self.range.contains(&position));
        let root_range = self.range.clone();
        Self::point_query_inner(
            self,
            position,
            &root_range,
            0.into(),
            |zelf, idx| {
                f(zelf, idx);
                zelf
            },
            Self::create_left_child,
            Self::create_right_child,
        );
    }

    #[allow(unused_must_use)]
    fn range_query(&self, range: Range<u64>, mut arg: impl FnMut(&Self, Self::NodeIdx)) {
        assert!(range.is_normal());
        assert!(self.range.is_superset(&range));
        // This returns a (must_use) closure, which we are happy to ignore
        Self::range_query_inner(
            self,
            &range,
            &self.range,
            0.into(),
            move |zelf, idx| {
                arg(zelf, idx);
                zelf
            },
            Self::do_not_create_children,
            Self::do_not_create_children,
        );
    }

    #[allow(unused_must_use)]
    fn range_query_mut(
        &mut self,
        range: Range<u64>,
        mut arg: impl FnMut(&mut Self, Self::NodeIdx),
    ) {
        assert!(range.is_normal());
        assert!(self.range.is_superset(&range));
        let root_range = self.range.clone();
        // This returns a (must_use) closure, which we are happy to ignore
        Self::range_query_inner(
            self,
            &range,
            &root_range,
            0.into(),
            move |zelf, idx| {
                arg(zelf, idx);
                zelf
            },
            Self::create_left_child,
            Self::create_right_child,
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::segment_tree::SegmentTree;

    use super::ImplicitSegmentTree;

    #[test]
    fn immutable_calls_do_not_create_nodes() {
        let tree = ImplicitSegmentTree::<(), _>::with_default(0..100);
        tree.point_query(0, |_, _| {});
        tree.range_query(0..100, |_, _| {});

        assert_eq!(tree.data.len(), 1);
    }
}
