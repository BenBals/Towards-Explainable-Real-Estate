use algorithms::calculate_environment::calculate_U_for_immos;
use common::immo::ImmoIdx;
use common::{BpResult, CostFunction, Dissimilarity, Immo, Trainable};
use typed_index_collections::TiVec;

/// A wrapper which creates a CostFunction based on a Dissimilarity.
/// when constructed using [with_immos] all U-values are calculated based on the given dissimilarity
/// uses a lookup-table of size `max_idx` where `max_idx` is the maximum idx value of all given immos.
#[derive(Debug, Clone)]
pub struct DissimilarityCostFunction<D: Dissimilarity> {
    dissimilarity: D,
    u_values: TiVec<ImmoIdx, f64>,
}

impl<D: Dissimilarity> DissimilarityCostFunction<D> {
    /// creates a new [DissimilarityCostFunction] and calculates U-values for immos
    /// Every Immo needs an ImmoIdx. The caller needs to ensure, that [set_immo_idxs] was called.
    /// # Returns
    /// Err if an immo has no ImmoIdx set
    /// # Panics
    /// if [calculate_U_for_immos] panics
    pub fn with_immos<'i>(dissimilarity: D, immos: impl IntoIterator<Item = &'i Immo>) -> Self {
        let collected_immo_refs: Vec<_> = immos.into_iter().collect();
        let u_values_map = calculate_U_for_immos(&dissimilarity, &collected_immo_refs);

        let max_idx: usize = collected_immo_refs
            .iter()
            .filter_map(|immo| immo.idx)
            .max()
            .unwrap_or_default()
            .into();

        let mut u_values: TiVec<ImmoIdx, f64> = (0..max_idx as i32 + 1).map(|_| 0.0).collect();

        for immo in collected_immo_refs {
            let idx = immo
                .idx
                .expect("All Immos need their idx set. Call set_immo_idxs before.");
            u_values[idx] = u_values_map[immo.id()]
        }

        Self {
            dissimilarity,
            u_values,
        }
    }

    /// creates a new [DissimilarityCostFunction] from the given dissimilarity.
    /// For every Immo, `immo.u` is used.
    /// Use this over [with_immos] if u-calculation is expensive or if you need exactly the values from the database
    /// Every Immo needs an ImmoIdx and u.
    /// The caller needs to ensure, that [set_immo_idxs] was called.
    /// # Returns
    /// Err if an immo has no ImmoIdx set or u not set
    pub fn with_immo_u_values<'i>(
        dissimilarity: D,
        immos: impl IntoIterator<Item = &'i Immo>,
    ) -> Self {
        let collected_immo_refs: Vec<_> = immos.into_iter().collect();
        let max_idx: usize = collected_immo_refs
            .iter()
            .filter_map(|immo| immo.idx)
            .max()
            .unwrap_or_default()
            .into();

        let mut u_values: TiVec<ImmoIdx, f64> = (0..max_idx as i32 + 1).map(|_| 0.0).collect();

        for immo in collected_immo_refs {
            let idx = immo
                .idx
                .expect("All Immos need their idx set. Call set_immo_idxs before.");
            u_values[idx] = immo.u.expect("All Immos need to have their u value set");
        }

        Self {
            dissimilarity,
            u_values,
        }
    }
}

impl<D: Dissimilarity> CostFunction for DissimilarityCostFunction<D> {
    fn cost(&self, a: &Immo, b: &Immo) -> f64 {
        self.dissimilarity.dissimilarity(a, b) + self.dissimilarity.dissimilarity(b, a)
            - self.u_values[a.idx.unwrap()]
            - self.u_values[b.idx.unwrap()]
    }
}

impl<D: Dissimilarity> Trainable for DissimilarityCostFunction<D> {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        self.dissimilarity.train(training_data)
    }
}

impl<D: Dissimilarity> Dissimilarity for DissimilarityCostFunction<D> {
    fn dissimilarity(&self, this: &Immo, other: &Immo) -> f64 {
        self.dissimilarity.dissimilarity(this, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use common::immo::{set_immo_idxs, ImmoBuilder};
    use dissimilarities::{ConstantDissimilarity, SqmPriceDissimilarity};
    use proptest::prelude::*;
    use test_helpers::full_immos;

    #[test]
    fn example_with_immos() {
        let mut immos = vec![
            ImmoBuilder::default()
                .marktwert(10.0)
                .wohnflaeche(1.0)
                .plane_location((0.0, 0.0))
                .build()
                .unwrap(),
            ImmoBuilder::default()
                .marktwert(1.0)
                .wohnflaeche(1.0)
                .plane_location((20.0, 0.0))
                .build()
                .unwrap(),
            ImmoBuilder::default()
                .marktwert(5.0)
                .wohnflaeche(1.0)
                .plane_location((10.0, 10.0))
                .build()
                .unwrap(),
        ];
        set_immo_idxs(immos.iter_mut());
        let dissimilarity =
            DissimilarityCostFunction::with_immos(SqmPriceDissimilarity, immos.iter());

        let weight_total = 1.0 / 400.0 + 1.0 / 200.0;
        assert_approx_eq!(
            dissimilarity.cost(&immos[0], &immos[1]),
            81.0 * 2.0
                - ((1.0 / 400.0) / weight_total * 81.0 + (1.0 / 200.0) / weight_total * 25.0)
                - ((1.0 / 400.0) / weight_total * 81.0 + (1.0 / 200.0) / weight_total * 16.0)
        );
    }

    proptest! {
        #[test]
        fn with_immo_u_values_u_values_are_correct(immos in full_immos(16)) {
            let dissimilarity = DissimilarityCostFunction::with_immo_u_values(
                ConstantDissimilarity::with(0.0),
                immos.iter(),
            );

            for immo1 in &immos {
                for immo2 in &immos {
                    prop_assert!(dissimilarity.dissimilarity(immo1, immo2).abs() < 1e-3);
                    prop_assert!(
                        (dissimilarity.cost(immo1, immo2) + immo1.u.unwrap() + immo2.u.unwrap()).abs() < 1e-3
                    );
                }
            }
        }

        #[test]
        fn with_immos_u_values_are_correct(mut immos in full_immos(16)) {
            for immo in &mut immos {
                immo.plane_location = Some((0.0, 0.0));
            }
            let dissimilarity = DissimilarityCostFunction::with_immos(
                ConstantDissimilarity::with(1.0),
                immos.iter(),
            );

            for immo1 in &immos {
                for immo2 in &immos {
                    if immo1 == immo2 {
                        continue;
                    }

                    prop_assert!((dissimilarity.dissimilarity(immo1, immo2) - 1.0).abs() < 1e-3);
                    prop_assert!(
                        dissimilarity.cost(immo1, immo2).abs() < 1e-3
                    );
                }
            }
        }
    }
}
