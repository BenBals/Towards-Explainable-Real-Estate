use itertools::Itertools;

use super::*;

impl<T: DcelNum, M> Dcel<T, M> {
    fn iter_faces(&self) -> impl Iterator<Item = (FaceIndex, &Face<T>)> {
        self.faces
            .iter_enumerated()
            .filter_map(|(index, opt)| Some(index).zip(opt.as_ref()))
    }

    fn iter_half_edges(&self) -> impl Iterator<Item = (HalfEdgeIndex, &HalfEdge)> {
        self.half_edges
            .iter_enumerated()
            .filter_map(|(index, opt)| Some(index).zip(opt.as_ref()))
    }

    fn validate_vertices_first_edge_matches(&self) -> bool {
        self.vertices.iter_enumerated().all(|(idx, vertex)| {
            let cur_valid = self[vertex.first_half_edge].from == idx;
            if !cur_valid {
                log::error!(
                    "First half edge of {:?} does not have from set correctly.",
                    idx
                );
            }
            cur_valid
        })
    }

    fn validate_twin_edges_are_reversed(&self) -> bool {
        self.iter_half_edges().all(|(index, _)| {
            let twin_index = self[index].twin;

            let cur_valid =
                self[index].from == self[twin_index].to && self[twin_index].from == self[index].to;
            if !cur_valid {
                log::error!(
                    "{:?} has {:?} as twin but vertices fo not match",
                    index,
                    twin_index
                );
            }
            cur_valid
        })
    }

    fn validate_next_is_set_correctly<'a>(
        &self,
        iter_fn: impl Fn(HalfEdgeIndex) -> Box<dyn Iterator<Item = HalfEdgeIndex> + 'a>,
        test_fn: impl Fn(&T) -> bool,
        name: &str,
    ) -> bool {
        self.iter_half_edges().all(|(index, _)| {
            let start_pos = &self[self[index].from].position;
            let slice: &[_] = &iter_fn(index)
                .map(|edge_idx| self[self[edge_idx].to].position.clone() - start_pos)
                .collect::<Vec<_>>();
            if let [a, b, c, ..] = slice {
                let cur_valid = test_fn(&a.cross(c).dot(b));
                if !cur_valid {
                    log::error!(
                        "Orientation of edges by {} are wrong for edge {:?}.",
                        name,
                        index
                    );
                }
                cur_valid
            } else {
                unreachable!()
            }
        })
    }

    fn validate_iter_edges_are_reversed_of_each_other(&self) -> bool {
        self.iter_half_edges().all(|(index, _)| {
            // append index to make them reversed of each other
            let counter_cw_edges = self
                .iter_half_edges_counter_clock_wise(index)
                .chain(Some(index));
            let cw_edges = self.iter_half_edges_clock_wise(index).chain(Some(index));
            let cur_valid = itertools::equal(
                counter_cw_edges.clone(),
                cw_edges.clone().collect::<Vec<_>>().into_iter().rev(),
            );
            if !cur_valid {
                log::error!(
                    "CW and CCW Edge iterations for {:?} are not the same.\nCCW: {:?}\nCW: {:?}",
                    index,
                    counter_cw_edges.collect::<Vec<_>>(),
                    cw_edges
                        .clone()
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>(),
                )
            }
            cur_valid
        })
    }

    fn validate_iter_edges_gives_all_edges_from_same_start_node<'a>(
        &self,
        iter_fn: impl Fn(HalfEdgeIndex) -> Box<dyn Iterator<Item = HalfEdgeIndex> + 'a>,
        name: &str,
    ) -> bool {
        let vertex_edge = {
            let mut vertex_edge: TiVec<VertexIndex, _> =
                (0..self.vertices.len()).map(|_| Vec::new()).collect();
            for (edge_idx, half_edge) in self.iter_half_edges() {
                vertex_edge[half_edge.from].push(edge_idx);
            }
            vertex_edge.iter_mut().for_each(|vec| vec.sort_unstable());
            vertex_edge
        };

        self.iter_half_edges().all(|(index, edge)| {
            let cur_edges = iter_fn(index)
                .sorted_unstable();
            let cur_valid = itertools::equal(
                &vertex_edge[edge.from],
                cur_edges.as_ref(),
            );
            if !cur_valid {
                log::error!("Iter {} for {:?} did not give the edges which start from the same vertex.\nStarting from Vertex:{:?}\nIterated: {:?}", name, index, vertex_edge[edge.from], cur_edges.collect::<Vec<_>>());
            }
            cur_valid
        })
    }

    fn validate_surrounding_edges_form_partition(&self) -> bool {
        let face_to_edge: TiVec<FaceIndex, _> = (0..self.faces.len())
            .map(FaceIndex)
            .map(|index| {
                if self.faces[index].is_some() {
                    self.iter_surrounding_half_edges(index).collect()
                } else {
                    Vec::new()
                }
            })
            .collect();

        self.iter_half_edges().all(|(index, _)| {
            let matching_faces = face_to_edge
                .iter_enumerated()
                .filter(|(_, edges)| edges.contains(&index));

            let cur_valid = matching_faces.clone().exactly_one().is_ok();
            if !cur_valid {
                log::error!(
                    "{:?} does not belong to exactly one face, but rather {:?}",
                    index,
                    matching_faces
                        .clone()
                        .map(|(face_idx, _)| face_idx)
                        .collect::<Vec<_>>()
                )
            }
            cur_valid
        })
    }

    fn validate_all_faces_surrounding_edges_form_closed_loop(&self) -> bool {
        self.iter_faces().all(|(face_index, _)| {
            let edges: Vec<_> = self.iter_surrounding_half_edges(face_index).collect();
            let mut valid = self[*edges.first().unwrap()].from == self[*edges.last().unwrap()].to;

            for (&prev_edge, &next_edge) in edges.iter().zip(edges.iter().skip(1)) {
                valid &= self[prev_edge].to == self[next_edge].from;
            }

            if !valid {
                log::error!(
                    "The edges of {:?} do not form a closed loop.\nEdges: {:#?}\nVertices:{:#?}",
                    face_index,
                    edges,
                    edges
                        .iter()
                        .map(|&edge_idx| self[edge_idx].from)
                        .collect::<Vec<_>>()
                );
            }
            valid
        })
    }

    fn validate_all_faces_vertices_lie_on_the_face(&self) -> bool {
        self.iter_faces().all(|(face_index, _)| {
            let dot_invariant = self[face_index]
                .outward_vector
                .dot(&self[self[self[face_index].start_half_edge].from].position);

            let valid = self
                .iter_surrounding_half_edges(face_index)
                .all(|edge_idx| {
                    let cur_dot = self[face_index]
                        .outward_vector
                        .dot(&self[self[edge_idx].from].position);
                    dot_invariant.dcel_eq(&cur_dot)
                });
            if !valid {
                log::error!(
                    "Not all vertices on Face {:?} are on the same plane",
                    face_index
                );
            }
            valid
        })
    }

    fn validate_all_faces_define_plane_of_support(&self) -> bool {
        self.iter_faces().all(|(face_index, face)| {
            let face_vector = &self[self[face.start_half_edge].from].position;
            let violating_vertices: Vec<_> = self
                .vertices
                .iter_enumerated()
                .filter_map(|(vertex_index, vertex)| {
                    if !(vertex.position.clone() - face_vector)
                        .dot(&face.outward_vector)
                        .dcel_non_positive()
                    {
                        Some(vertex_index)
                    } else {
                        None
                    }
                })
                .collect();

            let cur_valid = violating_vertices.is_empty();
            if !cur_valid {
                log::error!(
                    "{:?} is violated by vertices {:?}",
                    face_index,
                    violating_vertices
                );
            }
            cur_valid
        })
    }

    pub fn validate(&self) -> bool {
        let mut valid = true;

        valid &= self.validate_vertices_first_edge_matches();
        valid &= self.validate_twin_edges_are_reversed();

        valid &= self.validate_iter_edges_gives_all_edges_from_same_start_node(
            |index| Box::new(self.iter_half_edges_counter_clock_wise(index)),
            "CCW",
        );
        valid &= self.validate_iter_edges_gives_all_edges_from_same_start_node(
            |index| Box::new(self.iter_half_edges_clock_wise(index)),
            "CW",
        );
        valid &= self.validate_iter_edges_are_reversed_of_each_other();

        valid &= self.validate_next_is_set_correctly(
            |index| Box::new(self.iter_half_edges_counter_clock_wise(index)),
            T::dcel_non_positive,
            "CCW",
        );
        valid &= self.validate_next_is_set_correctly(
            |index| Box::new(self.iter_half_edges_clock_wise(index)),
            T::dcel_non_negative,
            "CW",
        );

        valid &= self.validate_all_faces_surrounding_edges_form_closed_loop();
        valid &= self.validate_all_faces_vertices_lie_on_the_face();
        valid &= self.validate_all_faces_define_plane_of_support();
        valid &= self.validate_surrounding_edges_form_partition();

        valid
    }
}

#[cfg(test)]
mod tests {
    use crate::distance_approximation::dcel::metrics::NoMetric;

    use super::*;
    use proptest::prelude::*;

    #[test]
    pub fn correctly_rotated_simplex_is_valid() {
        let dcel = Dcel::tetrahedron(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            NoMetric,
        )
        .expect("The points are found to be coplanar, but they are not");
        assert!(dcel.validate());
    }

    #[test]
    pub fn all_simplex_permutations_are_valid() {
        [
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
        ]
        .iter()
        .permutations(4)
        .for_each(|vec| {
            let cloned: &[_] = &vec.iter().map(|vector3| **vector3).collect::<Vec<_>>();
            if let [a, b, c, d] = *cloned {
                let dcel = Dcel::tetrahedron(a, b, c, d, NoMetric)
                    .expect("The points are found to be coplanar, but they are not");
                assert!(
                    dcel.validate(),
                    "Simplex with permutation is invalid\n{}\n{}\n{}\n{}\n{:#?}",
                    a,
                    b,
                    c,
                    d,
                    dcel,
                );
            } else {
                unreachable!()
            }
        });
    }

    #[test]
    pub fn adding_corner_to_half_cube_is_valid() {
        let mut dcel = Dcel::tetrahedron(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            NoMetric,
        )
        .expect("The points are found to be coplanar, but they are not");
        dcel.add_vertex(Vector3::new(1.0, 1.0, 0.0), None);
        assert!(dcel.validate());
    }

    #[test]
    pub fn adding_point_on_existing_face_is_none() {
        let mut dcel = Dcel::tetrahedron(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            NoMetric,
        )
        .expect("The points are found to be coplanar, but they are not");
        let vert_opt = dcel.add_vertex(Vector3::new(0.3, 0.3, 0.0), None);
        assert!(
            vert_opt.is_none(),
            "Vertex lies on existing face, but was inserted nonetheless"
        );
        assert!(dcel.validate());
    }

    #[test]
    pub fn adding_point_in_interior_is_none() {
        let mut dcel = Dcel::tetrahedron(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            NoMetric,
        )
        .expect("The points are found to be coplanar, but they are not");
        let vert_opt = dcel.add_vertex(Vector3::new(0.1, 0.1, 0.1), None);
        assert!(
            vert_opt.is_none(),
            "Vertex lies in the inside, but was inserted nonetheless"
        );
        assert!(dcel.validate());
    }

    fn check_iterative_insertion<T>(points: Vec<&'_ Vector3<T>>, can_be_coplanar: bool)
    where
        T: DcelNum + Clone,
    {
        let dcel_opt = Dcel::tetrahedron(
            points[0].clone(),
            points[1].clone(),
            points[2].clone(),
            points[3].clone(),
            NoMetric,
        );
        if dcel_opt.is_none() && can_be_coplanar {
            return;
        }

        let mut dcel = dcel_opt.expect("The points are found to be coplanar, but they are not");
        assert!(
            dcel.validate(),
            "Dcel from {:?} is invalid after adding first 4 points",
            points
        );
        for (index, &point) in points.iter().enumerate().skip(4) {
            let vert_opt = dcel.add_vertex(point.clone(), None);
            if !can_be_coplanar && vert_opt.is_none() {
                panic!("Points can't be coplanar but insertion failed");
            }
            assert!(
                dcel.validate(),
                "Dcel from {:?} is invalid after adding first {} points",
                points,
                index + 1,
            );
        }
    }

    #[test]
    pub fn creating_double_pyramid_is_valid_in_all_permutations() {
        // all corners of the shape
        // no 4 coplanar points
        let all_points: Vec<_> = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.5, 1.0, 0.0),
            Vector3::new(0.5, 0.5, 1.0),
            Vector3::new(0.5, 0.5, -1.0),
        ];

        all_points
            .iter()
            .permutations(all_points.len())
            .for_each(|points| check_iterative_insertion(points, false));
    }

    #[test]
    pub fn creating_cube_is_valid_in_all_non_coplanar_permutations() {
        let all_points: Vec<_> = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
        ];

        all_points
            .iter()
            .combinations(4)
            .for_each(|initial_points| {
                let points_to_insert = all_points
                    .iter()
                    .filter(|vector| !initial_points.contains(vector));
                for insertions in points_to_insert.permutations(all_points.len() - 4) {
                    let points: Vec<_> = initial_points
                        .iter()
                        .copied()
                        .chain(insertions.into_iter())
                        .collect();
                    check_iterative_insertion(points, true);
                }
            });
    }

    #[test]
    fn regression_points_on_sphere() {
        // this test checks previously failed since multiple edge deletions at a single vertex was broken

        // [0.12983056492942538, 0.9915361942007495, 0.0]
        // [-0.6508731757973073, 0.7591864784277493, -0.0]
        // [1.0, 0.0, 0.0]
        // [0.7887837512719832, 0.0, 0.6146708011035648]
        // [0.892979130880015, -0.45009806910580386, 0.0]
        // [-0.04430486637606562, -0.705356815285755, -0.7074664952811096]
        // manual simplification yielded
        let points = vec![
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(-1.0, 2.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 1.0),
            Vector3::new(2.0, -1.0, 0.0),
            Vector3::new(0.0, -2.0, -1.0),
        ];
        let point_refs: Vec<_> = points.iter().collect();

        check_iterative_insertion(point_refs, true);
    }

    prop_compose! {
        pub fn viable_3d_vector(max: f64)(
            a in max.recip()..max,
            b in max.recip()..max,
            c in max.recip()..max,
        ) -> Vector3<f64> {
            Vector3::new(a,b,c)
        }
    }

    prop_compose! {
        pub fn point_on_unit_sphere() (
            phi in 0.0..std::f64::consts::TAU,
            theta in 0.0..std::f64::consts::TAU,
        ) -> Vector3<f64> {
            let y = phi.sin();
            let x = phi.cos() * theta.cos();
            let z = phi.cos() * theta.sin();
            Vector3::new(x, y, z)
        }
    }

    proptest! {
        #[test]
        fn non_planar_tetrahedron_is_valid(
            a in viable_3d_vector(1e5),
            b in viable_3d_vector(1e5),
            c in viable_3d_vector(1e5),
            d in viable_3d_vector(1e5),
        ) {
            common::logging::init_test_logging();
            prop_assume!(!(b-a).cross(&(c-a)).dot(&(d-a)).dcel_eq(&0.0));
            let dcel = Dcel::tetrahedron(a,b,c,d, NoMetric).expect("The points are found to be coplanar, but they are not");
            prop_assert!(dcel.validate());
        }

        #[test]
        fn planar_tetrahedron_is_none(
            // use small values to escape f64 problems
            a in viable_3d_vector(10.0),
            b in viable_3d_vector(10.0),
            c in viable_3d_vector(10.0),
            ab_factor in -10f64..10f64,
            ac_factor in -10f64..10f64,
        ) {
            let d = a + (b - a) * ab_factor + (c - a) * ac_factor;
            let dcel = Dcel::tetrahedron(a,b,c,d, NoMetric);
            prop_assert!(dcel.is_none(), "Points are coplanar, but this was not found");
        }

        #[test]
        fn adding_points_on_sphere_is_valid(
            points in prop::collection::vec(point_on_unit_sphere(), 4..32)
        ) {
            let point_refs: Vec<_> = points.iter().collect();
            // pair wise different
            prop_assume!(
                point_refs
                    .iter()
                    .combinations(2)
                    .all(|points| !points[0].dcel_eq(points[1]))
            );

            check_iterative_insertion(point_refs, true);
        }
    }
}
