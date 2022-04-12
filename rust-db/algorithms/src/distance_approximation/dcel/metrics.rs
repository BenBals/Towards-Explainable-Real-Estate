use std::cell::RefCell;

use itertools::Itertools;

use super::{Dcel, DcelNum, VertexIndex};

/// This is a trait which is implemented by everything that can collect metrics about the dcel process.
pub trait DcelMetric: Sized {
    /// This function will be called by the dcel, when the insertion of a vertex is finished,
    /// but the cleanup is not done. This is a good point to collect metrics about the caused change.
    /// # Arguments
    /// * `dcel`: The dcel to collect metrics about.
    /// * `vertex_index`: The currently inserted vertex.
    fn end_add_vertex<T: DcelNum>(&self, dcel: &Dcel<T, Self>, vertex_index: VertexIndex);
}

#[derive(Debug, Clone, Copy)]
/// A Metric which just ignores everything.
pub struct NoMetric;

impl DcelMetric for NoMetric {
    fn end_add_vertex<T: DcelNum>(&self, _dcel: &Dcel<T, Self>, _vertex_index: VertexIndex) {}
}

#[derive(Debug, Clone, Default)]
struct ChangeMetricInner {
    added_half_edges: Vec<usize>,
    removed_half_edges: Vec<usize>,
    added_faces: Vec<usize>,
    removed_faces: Vec<usize>,
}

#[derive(Debug, Default)]
/// A Metric which records the change in the datastructure
pub struct ChangeMetric(RefCell<ChangeMetricInner>);

impl ChangeMetric {
    pub fn average_changed_half_edges(&self) -> f64 {
        let inner = self.0.borrow();
        let changed_half_edges = inner
            .added_half_edges
            .iter()
            .zip(&inner.removed_half_edges)
            .zip(&inner.added_faces)
            // exactly the added and removed half edges are changed as well as exactly one old edge adjacent to each new face
            .map(|((added_half_edges, removed_half_edges), added_faces)| {
                (added_half_edges + removed_half_edges + added_faces) as f64
            });
        let count = changed_half_edges.clone().count() as f64;
        changed_half_edges.sum::<f64>() / count
    }

    pub fn average_changed_faces(&self) -> f64 {
        let inner = self.0.borrow();
        let changed_faces = inner
            .added_faces
            .iter()
            .zip(&inner.removed_faces)
            .map(|(added, removed)| (added + removed) as f64);
        let count = changed_faces.clone().count() as f64;
        changed_faces.sum::<f64>() / count
    }
}

impl DcelMetric for ChangeMetric {
    fn end_add_vertex<T: DcelNum>(&self, dcel: &Dcel<T, Self>, vertex_index: VertexIndex) {
        let mut inner = self.0.borrow_mut();

        let added_vertex_edges = dcel
            .iter_half_edges_clock_wise(dcel[vertex_index].first_half_edge)
            .count();

        // every edge is either directed away from the newly created vertex or are a twin of this
        inner.added_half_edges.push(added_vertex_edges * 2);
        // there is a bijection between added faces and added edges from the new vertex
        inner.added_faces.push(added_vertex_edges);

        let removed_edges = dcel.half_edges_to_remove.iter().unique().count();
        inner.removed_half_edges.push(removed_edges);

        let removed_faces = dcel.faces_to_remove.iter().unique().count();
        inner.removed_faces.push(removed_faces);
    }
}
