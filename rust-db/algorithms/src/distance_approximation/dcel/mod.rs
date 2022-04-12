mod maths;
pub use maths::*;

pub mod metrics;

#[cfg(any(debug_assertions, test))]
mod validation;

use std::{
    iter,
    ops::{Index, IndexMut},
};

use derive_more::{From, Into};
use typed_index_collections::TiVec;

use self::metrics::DcelMetric;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
#[doc(hidden)]
pub struct VertexIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
#[doc(hidden)]
pub struct HalfEdgeIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
#[doc(hidden)]
pub struct FaceIndex(usize);

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct Vertex<T: DcelNum> {
    pub(super) position: Vector3<T>,
    pub(super) first_half_edge: HalfEdgeIndex,
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct HalfEdge {
    pub(super) from: VertexIndex,
    pub(super) to: VertexIndex,
    pub(super) next_counter_clock_wise: HalfEdgeIndex,
    pub(super) next_clock_wise: HalfEdgeIndex,
    pub(super) twin: HalfEdgeIndex,
    pub(super) face: FaceIndex,
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct Face<T: DcelNum> {
    pub(super) start_half_edge: HalfEdgeIndex,
    // this vector points in the same direction as the normal, but is NOT normalized
    pub(super) outward_vector: Vector3<T>,
}

#[derive(Debug, Clone)]
/// This is a data structure which can be used to efficiently represent 3d polyhedra.
pub struct Dcel<T: DcelNum, M> {
    pub(super) vertices: TiVec<VertexIndex, Vertex<T>>,
    pub(super) half_edges: TiVec<HalfEdgeIndex, Option<HalfEdge>>,
    pub(super) faces: TiVec<FaceIndex, Option<Face<T>>>,
    pub(super) none_half_edge_indices: Vec<HalfEdgeIndex>,
    pub(super) half_edges_to_remove: Vec<HalfEdgeIndex>,
    pub(super) none_face_indices: Vec<FaceIndex>,
    pub(super) faces_to_remove: Vec<FaceIndex>,
    pub metric: M,
}

impl<T: DcelNum> Face<T> {
    /// Given three triangle in ccw order this function computes the outward facing normal
    fn compute_outward_normal(a: &Vector3<T>, b: &Vector3<T>, c: &Vector3<T>) -> Vector3<T> {
        (a.clone() - b).cross(&(c.clone() - a)).normalize()
        // let outward = (a - b).cross(&(c - a))
        // outward.normalize()
    }

    fn test_side<M>(
        &self,
        point: &Vector3<T>,
        dcel: &Dcel<T, M>,
        test_fn: impl Fn(T) -> bool,
    ) -> bool {
        let point_on = &dcel[dcel[self.start_half_edge].from].position;
        let outward_distance = (point.clone() - point_on).dot(&self.outward_vector);
        test_fn(outward_distance)
    }

    pub fn is_on<M>(&self, point: &Vector3<T>, dcel: &Dcel<T, M>) -> bool {
        self.test_side(point, dcel, |dist| dist.dcel_eq(&T::zero()))
    }

    pub fn is_outside<M>(&self, point: &Vector3<T>, dcel: &Dcel<T, M>) -> bool {
        self.test_side(point, dcel, |dist| dist.dcel_greater(&T::zero()))
    }

    pub fn is_inside<M>(&self, point: &Vector3<T>, dcel: &Dcel<T, M>) -> bool {
        self.test_side(point, dcel, |dist| dist.dcel_less(&T::zero()))
    }
}

impl<T: DcelNum, M> Dcel<T, M> {
    pub fn iter_surrounding_half_edges(
        &self,
        face_idx: FaceIndex,
    ) -> impl Iterator<Item = HalfEdgeIndex> + '_ {
        let start = self[face_idx].start_half_edge;
        iter::successors(Some(start), move |&current| {
            let next = self[self[current].twin].next_clock_wise;
            if next != start {
                Some(next)
            } else {
                None
            }
        })
    }

    pub fn iter_half_edges_counter_clock_wise(
        &self,
        edge_idx: HalfEdgeIndex,
    ) -> impl Iterator<Item = HalfEdgeIndex> + Clone + '_ {
        iter::successors(Some(edge_idx), move |&current| {
            let next = self[current].next_counter_clock_wise;
            if next != edge_idx {
                Some(next)
            } else {
                None
            }
        })
    }

    pub fn iter_half_edges_clock_wise(
        &self,
        edge_idx: HalfEdgeIndex,
    ) -> impl Iterator<Item = HalfEdgeIndex> + Clone + '_ {
        iter::successors(Some(edge_idx), move |&current| {
            let next = self[current].next_clock_wise;
            if next != edge_idx {
                Some(next)
            } else {
                None
            }
        })
    }

    fn get_vacant_half_edge_entry(&mut self) -> HalfEdgeIndex {
        self.none_half_edge_indices
            .pop()
            .unwrap_or_else(|| self.half_edges.push_and_get_key(None))
    }

    fn get_vacant_face_entry(&mut self) -> FaceIndex {
        self.none_face_indices
            .pop()
            .unwrap_or_else(|| self.faces.push_and_get_key(None))
    }

    fn remove_half_edge_from_doubly_linked_pointers(&mut self, index: HalfEdgeIndex) {
        let next_cw = self[index].next_clock_wise;
        if next_cw != HalfEdgeIndex::INVALID {
            self[next_cw].next_counter_clock_wise = HalfEdgeIndex::INVALID;
        }

        let next_counter_cw = self[index].next_counter_clock_wise;
        if next_counter_cw != HalfEdgeIndex::INVALID {
            self[next_counter_cw].next_clock_wise = HalfEdgeIndex::INVALID;
        }
    }

    fn mark_half_edge_entry_to_remove(&mut self, index: HalfEdgeIndex) {
        self.half_edges_to_remove.push(index);
        self.remove_half_edge_from_doubly_linked_pointers(index);
    }

    fn mark_face_entry_to_remove(&mut self, index: FaceIndex) {
        self.faces_to_remove.push(index);
    }

    fn finalize_removes(&mut self) {
        for index in self.faces_to_remove.clone() {
            if self.faces[index].is_some() {
                self.faces[index] = None;
                self.none_face_indices.push(index);
            }
        }

        for index in self.half_edges_to_remove.clone() {
            if self.half_edges[index].is_some() {
                self.half_edges[index] = None;
                self.none_half_edge_indices.push(index);
            }
        }

        self.faces_to_remove.clear();
        self.half_edges_to_remove.clear();
    }
}

impl<T: DcelNum, M: DcelMetric> Dcel<T, M> {
    /// Constructs a Dcel from four points in 3d space
    /// # Returns
    /// None, if the points are all coplanar. Otherwise Some(dcel) with a correct dcel is returned.
    pub fn tetrahedron(
        a: Vector3<T>,
        b: Vector3<T>,
        c: Vector3<T>,
        d: Vector3<T>,
        metric: M,
    ) -> Option<Self> {
        let [a, b, c, d] = {
            let mut points = [a, b, c, d];
            let highest_index = {
                points
                    .iter()
                    .enumerate()
                    .max_by(|(_, l_vec), (_, r_vec)| l_vec[2].partial_cmp(&r_vec[2]).unwrap())
                    .unwrap()
                    .0
            };
            let points_len = points.len();
            points.swap(highest_index, points_len - 1);

            let [a, b, c, d] = &points;
            let abc_outward_normal = Face::compute_outward_normal(a, b, c);
            let abc_dist_to_d = (d.clone() - a).dot(&abc_outward_normal);
            if abc_dist_to_d.dcel_eq(&T::zero()) {
                return None;
            } else if !abc_dist_to_d.dcel_non_positive() {
                // bottom face is oriented wrongly
                points.swap(0, 1);
            }
            points
        };

        let faces = vec![
            Face {
                start_half_edge: 0.into(),
                outward_vector: Face::compute_outward_normal(&a, &b, &c),
            },
            Face {
                start_half_edge: 3.into(),
                outward_vector: Face::compute_outward_normal(&a, &d, &b),
            },
            Face {
                start_half_edge: 6.into(),
                outward_vector: Face::compute_outward_normal(&c, &d, &a),
            },
            Face {
                start_half_edge: 9.into(),
                outward_vector: Face::compute_outward_normal(&b, &d, &c),
            },
        ]
        .into_iter()
        .map(Some)
        .collect();

        let vertices = vec![
            Vertex {
                position: a,
                first_half_edge: 0.into(),
            },
            Vertex {
                position: b,
                first_half_edge: 1.into(),
            },
            Vertex {
                position: c,
                first_half_edge: 2.into(),
            },
            Vertex {
                position: d,
                first_half_edge: 8.into(),
            },
        ]
        .into_iter()
        .collect();

        let half_edges = vec![
            HalfEdge {
                // 0
                from: 0.into(),
                to: 1.into(),
                next_counter_clock_wise: 6.into(),
                next_clock_wise: 3.into(),
                face: 0.into(),
                twin: 5.into(),
            },
            HalfEdge {
                // 1
                from: 1.into(),
                to: 2.into(),
                next_counter_clock_wise: 5.into(),
                next_clock_wise: 9.into(),
                face: 0.into(),
                twin: 11.into(),
            },
            HalfEdge {
                // 2
                from: 2.into(),
                to: 0.into(),
                next_counter_clock_wise: 11.into(),
                next_clock_wise: 7.into(),
                face: 0.into(),
                twin: 6.into(),
            },
            HalfEdge {
                // 3
                from: 0.into(),
                to: 3.into(),
                next_counter_clock_wise: 0.into(),
                next_clock_wise: 6.into(),
                face: 1.into(),
                twin: 8.into(),
            },
            HalfEdge {
                // 4
                from: 3.into(),
                to: 1.into(),
                next_counter_clock_wise: 8.into(),
                next_clock_wise: 10.into(),
                face: 1.into(),
                twin: 9.into(),
            },
            HalfEdge {
                // 5
                from: 1.into(),
                to: 0.into(),
                next_counter_clock_wise: 9.into(),
                next_clock_wise: 1.into(),
                face: 1.into(),
                twin: 0.into(),
            },
            HalfEdge {
                // 6
                from: 0.into(),
                to: 2.into(),
                next_counter_clock_wise: 3.into(),
                next_clock_wise: 0.into(),
                face: 2.into(),
                twin: 2.into(),
            },
            HalfEdge {
                // 7
                from: 2.into(),
                to: 3.into(),
                next_counter_clock_wise: 2.into(),
                next_clock_wise: 11.into(),
                face: 2.into(),
                twin: 10.into(),
            },
            HalfEdge {
                // 8
                from: 3.into(),
                to: 0.into(),
                next_counter_clock_wise: 10.into(),
                next_clock_wise: 4.into(),
                face: 2.into(),
                twin: 3.into(),
            },
            HalfEdge {
                // 9
                from: 1.into(),
                to: 3.into(),
                next_counter_clock_wise: 1.into(),
                next_clock_wise: 5.into(),
                face: 3.into(),
                twin: 4.into(),
            },
            HalfEdge {
                // 10
                from: 3.into(),
                to: 2.into(),
                next_counter_clock_wise: 4.into(),
                next_clock_wise: 8.into(),
                face: 3.into(),
                twin: 7.into(),
            },
            HalfEdge {
                // 11
                from: 2.into(),
                to: 1.into(),
                next_counter_clock_wise: 7.into(),
                next_clock_wise: 2.into(),
                face: 3.into(),
                twin: 1.into(),
            },
        ]
        .into_iter()
        .map(Some)
        .collect();

        Some(Self {
            vertices,
            half_edges,
            faces,
            none_half_edge_indices: Vec::new(),
            half_edges_to_remove: Vec::new(),
            none_face_indices: Vec::new(),
            faces_to_remove: Vec::new(),
            metric,
        })
    }

    // face of across_edge will be set to new edge and previous face will be deleted
    // twins will not be set
    // only (edge from across_edge.to to inner) will be included in already existing doubly connected lists
    fn add_triangle_across(
        &mut self,
        inner: VertexIndex,
        across_edge: HalfEdgeIndex,
    ) -> (
        HalfEdgeIndex, // edge from across_edge.to to inner
        HalfEdgeIndex, // edge from inner to across_edge.from
    ) {
        let across_twin = self[across_edge].twin;

        let to_inner = self.get_vacant_half_edge_entry();
        let inner_from = self.get_vacant_half_edge_entry();

        let face = self.get_vacant_face_entry();

        // create all data
        self.faces[face] = Some(Face {
            start_half_edge: across_edge,
            outward_vector: Face::compute_outward_normal(
                &self[self[across_edge].from].position,
                &self[self[across_edge].to].position,
                &self[inner].position,
            ),
        });

        self.half_edges[to_inner] = Some(HalfEdge {
            from: self[across_edge].to,
            to: inner,
            next_counter_clock_wise: across_twin,
            next_clock_wise: self[across_twin].next_clock_wise,
            twin: HalfEdgeIndex::INVALID,
            face,
        });

        self.half_edges[inner_from] = Some(HalfEdge {
            from: inner,
            to: self[across_edge].from,
            next_counter_clock_wise: HalfEdgeIndex::INVALID,
            next_clock_wise: HalfEdgeIndex::INVALID,
            twin: HalfEdgeIndex::INVALID,
            face,
        });

        // remove old data
        self.mark_face_entry_to_remove(self[across_edge].face);

        // update existing data
        {
            // insert to_inner into doubly connected list
            let next_cw_across_twin = self[self[across_edge].twin].next_clock_wise;
            self[next_cw_across_twin].next_counter_clock_wise = to_inner;
            self[across_twin].next_clock_wise = to_inner;

            // set face of across_edge
            self[across_edge].face = face;
        }

        (to_inner, inner_from)
    }

    fn find_initial_vertex(&self, new_position: &Vector3<T>) -> Option<VertexIndex> {
        self.faces
            .iter()
            .filter_map(|opt| {
                opt.clone()
                    .filter(|face| face.is_outside(new_position, self))
                    .map(|face| self[face.start_half_edge].from)
            })
            .next()
    }

    fn go_clock_wise_while_and_delete(
        &mut self,
        mut cur_edge: HalfEdgeIndex,
        go_ahead: impl Fn(HalfEdgeIndex, &Self) -> bool,
    ) -> HalfEdgeIndex {
        let vertex_idx = self[cur_edge].from;

        // find next at boundary and delete all interior
        while go_ahead(cur_edge, self) {
            let prev_edge = cur_edge;
            cur_edge = self[cur_edge].next_clock_wise;
            self.mark_face_entry_to_remove(self[prev_edge].face);
            self.mark_half_edge_entry_to_remove(prev_edge);

            // prevent dangling pointer
            self[vertex_idx].first_half_edge = cur_edge;
        }
        cur_edge
    }

    pub fn add_vertex(
        &mut self,
        new_position: Vector3<T>,
        start_vertex: Option<VertexIndex>,
    ) -> Option<VertexIndex> {
        let is_at_boundary = |cur_idx: HalfEdgeIndex, dcel: &Self| -> bool {
            let own_face = &dcel[dcel[cur_idx].face];
            let next_face = &dcel[dcel[dcel[cur_idx].next_clock_wise].face];
            own_face.is_outside(&new_position, dcel) && !next_face.is_outside(&new_position, dcel)
        };

        let start_vertex = start_vertex.or_else(|| self.find_initial_vertex(&new_position))?;

        let start_edge = self
            .iter_half_edges_counter_clock_wise(self[start_vertex].first_half_edge)
            .find(|&edge| is_at_boundary(edge, self))?;

        let inner = self.vertices.push_and_get_key(Vertex {
            position: new_position.clone(),
            first_half_edge: HalfEdgeIndex::INVALID,
        });

        // create first triangle
        let (initial_to_inner, initial_inner_from) = self.add_triangle_across(inner, start_edge);
        self[inner].first_half_edge = initial_inner_from;

        // walk along boundary
        let mut prev_to_inner = initial_to_inner;
        let mut prev_inner_from = initial_inner_from;
        while self[prev_to_inner].from != start_vertex {
            let cur_edge = self.go_clock_wise_while_and_delete(
                self[prev_to_inner].next_clock_wise,
                |edge, dcel| !is_at_boundary(edge, dcel),
            );

            let (cur_to_inner, cur_inner_from) = self.add_triangle_across(inner, cur_edge);

            // link inner edges
            self[cur_inner_from].next_clock_wise = prev_inner_from;
            self[prev_inner_from].next_counter_clock_wise = cur_inner_from;

            // set twins
            self[cur_inner_from].twin = prev_to_inner;
            self[prev_to_inner].twin = cur_inner_from;

            // link outer edges
            self[cur_edge].next_counter_clock_wise = prev_to_inner;
            self[prev_to_inner].next_clock_wise = cur_edge;

            // goto next edge
            prev_to_inner = cur_to_inner;
            prev_inner_from = cur_inner_from;
        }

        // cleanup edges at seam vertex
        {
            self.go_clock_wise_while_and_delete(self[prev_to_inner].next_clock_wise, |edge, _| {
                edge != start_edge
            });
            self[start_edge].next_counter_clock_wise = prev_to_inner;
            self[prev_to_inner].next_clock_wise = start_edge;
        }

        // finalize newly created edges
        {
            // link beginning and end
            self[initial_inner_from].next_clock_wise = prev_inner_from;
            self[prev_inner_from].next_counter_clock_wise = initial_inner_from;

            // set twins
            self[initial_inner_from].twin = prev_to_inner;
            self[prev_to_inner].twin = initial_inner_from;
        }

        self.metric.end_add_vertex(self, inner);
        self.finalize_removes();

        Some(inner)
    }
}

impl HalfEdgeIndex {
    const INVALID: Self = Self(usize::MAX);
}

impl<T: DcelNum, M> Index<VertexIndex> for Dcel<T, M> {
    type Output = Vertex<T>;

    fn index(&self, index: VertexIndex) -> &Self::Output {
        &self.vertices[index]
    }
}

impl<T: DcelNum, M> IndexMut<VertexIndex> for Dcel<T, M> {
    fn index_mut(&mut self, index: VertexIndex) -> &mut Self::Output {
        &mut self.vertices[index]
    }
}

impl<T: DcelNum, M> Index<HalfEdgeIndex> for Dcel<T, M> {
    type Output = HalfEdge;

    fn index(&self, index: HalfEdgeIndex) -> &Self::Output {
        self.half_edges[index].as_ref().unwrap()
    }
}

impl<T: DcelNum, M> IndexMut<HalfEdgeIndex> for Dcel<T, M> {
    fn index_mut(&mut self, index: HalfEdgeIndex) -> &mut Self::Output {
        self.half_edges[index].as_mut().unwrap()
    }
}

impl<T: DcelNum, M> Index<FaceIndex> for Dcel<T, M> {
    type Output = Face<T>;

    fn index(&self, index: FaceIndex) -> &Self::Output {
        self.faces[index].as_ref().unwrap()
    }
}

impl<T: DcelNum, M> IndexMut<FaceIndex> for Dcel<T, M> {
    fn index_mut(&mut self, index: FaceIndex) -> &mut Self::Output {
        self.faces[index].as_mut().unwrap()
    }
}
