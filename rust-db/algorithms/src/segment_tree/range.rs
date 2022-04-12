use std::ops::Range;

pub trait RangeExt {
    fn is_splittable(&self) -> bool;

    fn is_empty(&self) -> bool;

    /// A range is considered normal if `range.start <= range.end` holds.
    fn is_normal(&self) -> bool;

    fn split(&self) -> Option<(Self, Self)>
    where
        Self: Sized;

    fn intersects(&self, rhs: &Self) -> bool;

    fn is_superset(&self, other: &Self) -> bool;

    /// This restricts self, to not extend beyond other.
    fn clamp_by(&mut self, other: &Self);
}

impl RangeExt for Range<u64> {
    fn is_splittable(&self) -> bool {
        self.end - self.start > 1
    }

    fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    fn is_normal(&self) -> bool {
        self.start <= self.end
    }

    fn split(&self) -> Option<(Self, Self)>
    where
        Self: Sized,
    {
        if self.is_splittable() {
            let mid = self.start + (self.end - self.start) / 2;
            Some((self.start..mid, mid..self.end))
        } else {
            None
        }
    }

    fn intersects(&self, rhs: &Self) -> bool {
        if self.is_empty() || rhs.is_empty() {
            return false;
        }

        if self.start <= rhs.start {
            self.end > rhs.start
        } else {
            rhs.end > self.start
        }
    }

    fn is_superset(&self, other: &Self) -> bool {
        other.is_empty() || (self.start <= other.start && other.end <= self.end)
    }

    fn clamp_by(&mut self, other: &Self) {
        self.start = self.start.max(self.end.min(other.start));
        self.end = self.end.min(self.start.max(other.end));
    }
}

#[cfg(test)]
pub mod tests {
    use super::RangeExt;
    use std::ops::Range;

    use proptest::prelude::*;

    prop_compose! {
        fn normal_range()(a in prop::num::u64::ANY,
                          b in prop::num::u64::ANY)
            -> Range<u64> {
            a.min(b)..a.max(b)
        }
    }

    prop_compose! {
        // this might be degenerate
        fn empty_range()(a in prop::num::u64::ANY,
                         b in prop::num::u64::ANY)
            -> Range<u64> {
            a.max(b)..a.min(b)
        }
    }

    prop_compose! {

        fn range()(a in prop::num::u64::ANY,
                          b in prop::num::u64::ANY)
            -> Range<u64> {
            a..b
        }
    }

    proptest! {
        #[test]
        fn clamp_by_lets_ranges_stay_normal(mut l in normal_range(), r in range()) {
            l.clamp_by(&r);
            prop_assert!(l.is_normal());
        }

        #[test]
        fn clamp_by_with_degenerate_range_leaves_normal_empty_range(mut l in normal_range(), r in empty_range()) {
            l.clamp_by(&r);
            prop_assert!(l.is_normal());
            prop_assert!(l.is_empty());
        }

        #[test]
        fn clamp_by_does_not_unneccesarily_make_range_empty(mut l in range(), r in range()) {
            let orig = l.clone();
            l.clamp_by(&r);
            prop_assume!(l.is_empty());

            prop_assert!(orig.is_empty() || r.is_empty() || !orig.intersects(&r));
        }

        #[test]
        fn clamp_by_leaves_subset(mut l in range(), r in range()) {
            let orig = l.clone();
            l.clamp_by(&r);
            prop_assert!(orig.is_superset(&l));
        }

        #[test]
        fn clamp_by_sets_borders_correct_on_normal_intersecting_range(mut l in normal_range(), r in normal_range()) {
            prop_assume!(l.intersects(&r));

            let orig = l.clone();
            l.clamp_by(&r);

            if orig.contains(&r.start) {
                prop_assert_eq!(l.start, r.start);
            }
            if r.contains(&orig.start) {
                prop_assert_eq!(orig.start, l.start);
            }

            if orig.contains(&r.end) {
                prop_assert_eq!(l.end, r.end);
            }
            if r.contains(&orig.end) {
                prop_assert_eq!(orig.end, l.end);
            }
        }

        #[test]
        fn clamp_by_leaves_empty_normal_when_not_intersecting(mut l in normal_range(), r in normal_range()) {
            prop_assume!(!l.intersects(&r));

            l.clamp_by(&r);
            prop_assert!(l.is_normal());
            prop_assert!(l.is_empty());
        }

        #[test]
        fn nothing_intersects_empty_range(l in range(), r in empty_range()) {
            prop_assert!(!l.intersects(&r));
        }

        #[test]
        fn every_thing_is_superset_of_empty_range(l in range(), r in empty_range()) {
            prop_assert!(l.is_superset(&r));
        }
    }
}
