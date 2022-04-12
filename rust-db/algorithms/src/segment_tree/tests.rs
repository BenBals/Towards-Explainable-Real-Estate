macro_rules! generic_segment_tree_tests {
    ($ident: ident, $type_vec : ty) => {
        #[allow(clippy::reversed_empty_ranges)]
        mod $ident {
            use super::super::SegmentTree;
            use proptest::proptest;

            #[test]
            #[should_panic]
            fn new_left_greater_than_right_panics() {
                <$type_vec>::with_default(100..50);
            }

            #[test]
            #[should_panic]
            fn new_left_equal_right_panics() {
                <$type_vec>::with_default(100..100);
            }

            #[test]
            fn new_left_less_than_right_works() {
                <$type_vec>::with_default(100..150);
            }

            #[test]
            fn invalid_range_query_panics() {
                let tree = <$type_vec>::with_default(10..100);
                let left_bigger_than_right =
                    std::panic::catch_unwind(|| tree.range_query(20..15, |_, _| {}));
                let left_outside_range = std::panic::catch_unwind(|| {
                    tree.range_query((tree.borders().start - 1)..15, |_, _| {})
                });
                let right_outside_range = std::panic::catch_unwind(|| {
                    tree.range_query(10..(tree.borders().end + 1), |_, _| {})
                });
                assert!(left_bigger_than_right.is_err());
                assert!(left_outside_range.is_err());
                assert!(right_outside_range.is_err());
            }

            proptest! {
                #[test]
                fn range_query_point_update(left in 0..u64::MAX / 2, len in 1..128u64) {
                    let max = left + len;
                    let mut tree = <$type_vec>::with_default(left..max);
                    for i in left..max {
                        tree.point_query_mut(i as u64, |tree, idx| tree[idx].push(i));
                    }

                    for l in left..max {
                        for r in l..max {
                            let mut queried = Vec::new();
                            tree.range_query(l..r, |tree, idx| {
                                queried.extend(tree[idx].iter().copied());
                            });
                            queried.sort_unstable();
                            assert_eq!(queried, (l..r).collect::<Vec<_>>());
                        }
                    }
                }

                #[test]
                fn range_update_point_query(left in (0..u64::MAX / 2), len in 1..128u64) {
                    let max = left + len;
                    let mut tree = <$type_vec>::with_default(left..max);
                    for i in left..max {
                        tree.range_query_mut(i..max, |tree, idx| tree[idx].push(i));
                    }

                    for i in left..max {
                        let mut queried = Vec::new();
                        tree.point_query(i as u64, |tree, idx| {
                            queried.extend(tree[idx].iter().copied());
                        });
                        queried.sort_unstable();
                        assert_eq!(queried, (left..=i).collect::<Vec<_>>());
                    }
                }
            }
        }
    };
}

generic_segment_tree_tests!(
    explicit,
    crate::segment_tree::ExplicitSegmentTree::<Vec<u64>>
);
generic_segment_tree_tests!(
    implicit,
    crate::segment_tree::ImplicitSegmentTree::<Vec<u64>, fn() -> Vec<u64>>
);
