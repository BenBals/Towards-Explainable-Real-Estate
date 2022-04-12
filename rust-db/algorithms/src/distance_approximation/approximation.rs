use num::{rational::Ratio, Integer};

use crate::distance_approximation::dcel::{metrics::*, Dcel};

use super::dcel::{DcelNum, Vector3};

#[derive(Debug, Clone)]
enum InvertedQuadraticFunction {
    Quadratic {
        // 1 / a
        multiplier: f64,
        // - b / (2a)
        delta: f64,
        // c / a
        offset: f64,
        right_arc: bool,
    },
    Linear {
        n: f64,
        m: f64,
    },
}

impl InvertedQuadraticFunction {
    // left_arc, right_arc, middle
    pub fn with(a: f64, b: f64, c: f64) -> (Self, Self, f64) {
        if a.abs() >= 1e-9 {
            let multiplier = a.recip();
            let delta = -b * 0.5 * multiplier;
            let offset = c * multiplier;
            (
                Self::Quadratic {
                    multiplier,
                    delta,
                    offset,
                    right_arc: false,
                },
                Self::Quadratic {
                    multiplier,
                    delta,
                    offset,
                    right_arc: true,
                },
                delta,
            )
        } else {
            let inverse = Self::Linear { m: b, n: c };
            (inverse.clone(), inverse, 0.0)
        }
    }

    fn base_multiplier(&self) -> f64 {
        match self {
            Self::Quadratic {
                right_arc: true, ..
            } => 1.0,
            Self::Quadratic { .. } => -1.0,
            _ => 0.0,
        }
    }

    pub fn x_value_for(&self, y_value: f64) -> Option<f64> {
        match &self {
            Self::Quadratic {
                multiplier,
                delta,
                offset,
                ..
            } => {
                let radicand = y_value * multiplier + delta.powi(2) - offset;
                if radicand >= 0.0 {
                    Some((radicand).sqrt() * self.base_multiplier() + delta)
                } else if radicand.abs() < 1e-6 {
                    Some(*delta)
                } else {
                    None
                }
            }
            Self::Linear { n, m } => Some((y_value - n) / m),
        }
    }
}

#[derive(Clone, Debug)]
/// An approximation for a set of points by a piecewise quadratic function
pub struct QuadraticApproximation {
    // end of segment (exclusive)
    function_segments: Vec<(u64, InvertedQuadraticFunction)>,
}

trait ApproximationNum: DcelNum {
    fn map_2d_to_3d(x: f64, y: usize) -> Vector3<Self>;
}

impl ApproximationNum for f64 {
    fn map_2d_to_3d(x: f64, y: usize) -> Vector3<Self> {
        let y = y as f64;
        Vector3::new(x * x / y, x / y, 1.0 / y)
    }
}

impl<N> ApproximationNum for Ratio<N>
where
    N: Clone + Integer + From<u64>,
    Ratio<N>: DcelNum,
{
    fn map_2d_to_3d(x: f64, y: usize) -> Vector3<Self> {
        let x = N::from(x.ceil() as u64);
        let y = N::from(y as u64);
        Vector3::new(
            Self::new(x.clone() * x.clone(), y.clone()),
            Self::new(x, y.clone()),
            Self::new(1.into(), y),
        )
    }
}

impl QuadraticApproximation {
    // Maps 2d points separable by a quadratic function to 3d separable by a plane
    fn estimation_point_to_3d_space<N: ApproximationNum>(nth: usize, distance: f64) -> Vector3<N> {
        N::map_2d_to_3d(distance, nth + 1)
    }

    // Checks if the three vectors are colinear.
    fn are_points_colinear<N: DcelNum>(a: &Vector3<N>, b: &Vector3<N>, c: &Vector3<N>) -> bool {
        (b.clone() - a)
            .cross(&(c.clone() - a))
            .dcel_eq(&Default::default())
    }

    fn from_distances_with_overestimation_threshold_inner<N: ApproximationNum>(
        distances: Vec<f64>,
        _threshold: f64,
    ) -> Self {
        if distances.len() < 4 {
            // for now at least
            return Self {
                function_segments: Vec::new(),
            };
        }

        let mut initial_points: Vec<(usize, Vector3<N>)> = Vec::new();
        for (index, &dist) in distances.iter().enumerate() {
            if initial_points.len() == 3 {
                break;
            }

            let cur_3d_point = Self::estimation_point_to_3d_space(index, dist);
            let entry = (index, cur_3d_point);
            if initial_points.len() == 2
                && Self::are_points_colinear(&initial_points[0].1, &initial_points[1].1, &entry.1)
            {
                initial_points[1] = entry;
            } else {
                initial_points.push(entry);
            }
        }

        let slice: &[_] = &initial_points;
        if let [(_, a), (_, b), (c_index, c)] = slice {
            // there is at most one face containing the first 3 points as they are not colinear
            // therefore 2 rounds should be enough, but floats need more
            let d_index = c_index + 1;
            let (d_perturbation, mut dcel) = (0..5)
                .filter_map(|perturbation| {
                    Some(perturbation).zip(Dcel::tetrahedron(
                        a.clone(),
                        b.clone(),
                        c.clone(),
                        Self::estimation_point_to_3d_space(
                            d_index,
                            distances[d_index] + perturbation as f64,
                        ),
                        ChangeMetric::default(),
                    ))
                })
                .next()
                .unwrap();

            // todo change
            // this is just a stub to test this
            let mut last_point = None;
            let mut last_dist = distances[d_index] + d_perturbation as f64;
            for (cur_index, cur_dist) in distances.into_iter().enumerate().skip(d_index + 1) {
                let cur_dist = last_dist.max(cur_dist);
                let cur_point = Self::estimation_point_to_3d_space(cur_index, cur_dist);
                last_point = dcel.add_vertex(cur_point, last_point).or(last_point);
                last_dist = cur_dist;
            }

            eprintln!(
                "{} {} {}",
                dcel.metric.average_changed_faces(),
                dcel.metric.average_changed_half_edges(),
                dcel.vertices.len()
            );

            Self {
                function_segments: Vec::new(),
            }
        } else {
            todo!()
        }
    }

    /// Creates a [QuadraticApproximation] from the given distances.
    /// For each query point the number of distances within the approximation is at most (1+threshold)
    /// times the number of actual distances at most that of the query point.
    pub fn from_distances_with_overestimation_threshold(
        distances: Vec<f64>,
        _threshold: f64,
    ) -> Self {
        Self::from_distances_with_overestimation_threshold_inner::<f64>(distances, _threshold)
    }

    /// Gives the approximate distance to the k-th nearest neighbor.
    /// This approximation always gives an upper bound.
    pub fn approximate_knn_distance(&self, k: u64) -> Option<f64> {
        self.function_segments
            .last()
            .filter(|(last, _)| *last >= k)
            .map(|_| {
                let matching_index = match self
                    .function_segments
                    .binary_search_by_key(&k, |(end, _)| *end)
                {
                    Ok(index) => index - 1,
                    Err(index) => index,
                };
                self.function_segments[matching_index]
                    .1
                    .x_value_for(k as f64)
                    .unwrap()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::super::dcel::DcelNum;
    use super::*;
    use proptest::prelude::*;

    fn low_precision_eq(a: f64, b: f64) -> bool {
        // use own epsilon since this problem has really bad fp behavior
        const EPS: f64 = 1e-3;

        if (a - b).abs() < EPS {
            true
        } else if a.abs().max(b.abs()) >= 1.0 {
            (a - b).abs() / a < EPS || (a - b).abs() / b < EPS
        } else {
            false
        }
    }

    proptest! {
        #[test]
        fn middle_is_actually_the_middle(
            a in -1e6f64..1e6f64,
            b in -1e6f64..1e6f64,
            c in -1e6f64..1e6f64,
        ) {
            prop_assume!(!a.dcel_eq(&0.0));

            let (left, right, middle) = InvertedQuadraticFunction::with(a, b, c);
            let ym = a * middle* middle + b * middle + c;
            let left_x = left.x_value_for(ym).unwrap();
            let right_x = right.x_value_for(ym).unwrap();

            prop_assert!(
                low_precision_eq(left_x, middle) && low_precision_eq(middle, right_x),
                "All of left inverted middle {}, middle {}, right inverted middle {} should be equal",
                left_x,
                middle,
                right_x
            );
        }

        #[test]
        fn inverted_linear_function_is_actually_inverse(
            b in -1e6f64..1e6f64,
            c in -1e6f64..1e6f64,
            y in -1e6f64..1e6f64,
        ) {
            prop_assume!(!b.dcel_eq(&0.0));

            let (left, right, _) = InvertedQuadraticFunction::with(0.0, b, c);
            for arc in &[left, right] {
                let x_value = arc.x_value_for(y).unwrap();
                let y_value = b * x_value + c;
                prop_assert!(
                    low_precision_eq(y_value, y),
                    "given y {} should be equal to calculated y {}",
                    y,
                    y_value
                );
            }
        }

        #[test]
        fn inverted_quadratic_function_is_actually_inverse(
            a in -1e6f64..1e6f64,
            b in -1e6f64..1e6f64,
            c in -1e6f64..1e6f64,
            y in -1e6f64..1e6f64,
        ) {
            prop_assume!(!a.dcel_eq(&0.0));

            let xm = - b/ (2.0 * a);
            let ym = a * xm * xm + b * xm + c;

            let (left, right, _) = InvertedQuadraticFunction::with(a, b, c);
            if a.signum().dcel_eq(&(y - ym).signum()) {
                for arc in &[left, right] {
                    let x_value = arc.x_value_for(y).unwrap();
                    let y_value = a * x_value * x_value + b * x_value + c;
                    prop_assert!(
                        low_precision_eq(y_value, y),
                        "given y {} should be equal to calculated y {}",
                        y,
                        y_value
                    );
                }
            } else {
                for arc in &[left, right] {
                    prop_assert!(
                        arc.x_value_for(y).is_none(),
                        "Gave x_value when there should be none"
                    );
                }
            }
        }
    }
}
