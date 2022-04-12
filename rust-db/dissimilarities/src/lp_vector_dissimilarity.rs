//! Structs around ManhattanVectorDissimilarities for comparing two Immos
use common::immo::META_DATA_COUNT;
use common::Dissimilarity;
use common::{BpResult, Immo, Trainable};
use serde::Deserialize;

/// This dissimilarity metric takes a vector of all non-price information about the two [Immo]s and
/// calculates the `L_p`-Distance:
/// `p = 1`, which is the default, will give you Manhattandistance, `p = 2` is euclidean distance.
/// The values are taken from [Immo::meta_data_array] and can additionally be weighted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LpVectorDissimilarity {
    exponent: f64,
}

impl LpVectorDissimilarity {
    /// Creates a new [LpVectorDissimilarity] with the given exponent
    pub fn with_exponent(exponent: f64) -> Self {
        assert!(
            (0.1..=10.0).contains(&exponent),
            "exponent has to be between 0.1 and 10 to avoid accidental infinity as result"
        );
        Self { exponent }
    }

    /// returns a [LpVectorDissimilarity] with exponent `p = 1.0`
    pub fn manhattan() -> Self {
        Self::with_exponent(1.0)
    }
}

impl Default for LpVectorDissimilarity {
    fn default() -> Self {
        Self::manhattan()
    }
}

impl Trainable for LpVectorDissimilarity {}

impl Dissimilarity for LpVectorDissimilarity {
    fn dissimilarity(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        immo1
            .meta_data_array()
            .iter()
            .zip(immo2.meta_data_array().iter())
            .map(|options| match options {
                (Some(value1), Some(value2)) => (value1 - value2).abs().powf(self.exponent),
                _ => 0.0,
            })
            .sum::<f64>()
            .powf(1.0 / self.exponent)
    }
}

/// a trainable and more general version of [LpVectorDissimilarity] which scales all values between 0 and 1
/// It uses an exponent `p`, see [Wikipedia](https://en.wikipedia.org/wiki/Lp_space).
/// `p = 1`, which is the default, will give you Manhattandistance, `p = 2` is euclidean distance.
/// The values are taken from [Immo::meta_data_array] and can additionally be weighted.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalingLpVectorDissimilarity {
    /// The weights for [Immo::meta_data_array], these are applied after scaling and before exponentiation
    pub weights: [f64; META_DATA_COUNT],
    /// the exponent `p` for L_p this is applied after scaling and weighting
    pub exponent: f64,
    inner: InnerScalingLpVectorDissimilarity,
}

#[derive(Debug, Clone, PartialEq)]
enum InnerScalingLpVectorDissimilarity {
    Untrained,
    Trained {
        meta_data_ranges: [(f64, f64); META_DATA_COUNT],
    },
}

impl Default for InnerScalingLpVectorDissimilarity {
    fn default() -> Self {
        Self::Untrained
    }
}

/// Must always match config/dully/types.dhall
#[derive(Clone, Debug, Deserialize)]
pub struct ScalingLpVectorDissimilarityConfig {
    weights: [f64; META_DATA_COUNT],
    exponent: f64,
}

impl ScalingLpVectorDissimilarity {
    /// Creates a new dissimilarity with all weights and exponent equal to 1.0
    pub fn new() -> Self {
        Self::with_exponent(1.0)
    }

    /// Creates a new dissimilarity with given weights and exponent 1
    pub fn with_weights(weights: [f64; META_DATA_COUNT]) -> Self {
        Self::with_weights_and_exponent(weights, 1.0)
    }

    /// Creates a new dissimilarity with the given exponent where all weights are 1
    pub fn with_exponent(exponent: f64) -> Self {
        Self::with_weights_and_exponent([1.0; META_DATA_COUNT], exponent)
    }

    /// Creates a new dissimilarity with given weights and exponent
    pub fn with_weights_and_exponent(weights: [f64; META_DATA_COUNT], exponent: f64) -> Self {
        assert!(
            (0.1..=10.0).contains(&exponent),
            "exponent has to be between 0.1 and 10 to avoid accidental infinity as result"
        );
        Self {
            inner: InnerScalingLpVectorDissimilarity::Untrained,
            weights,
            exponent,
        }
    }

    /// Import weights and exponent from config file.
    pub fn from_config(config: &ScalingLpVectorDissimilarityConfig) -> BpResult<Self> {
        Ok(Self::with_weights_and_exponent(
            config.weights,
            config.exponent,
        ))
    }

    fn scale(&self, value: f64, idx: usize) -> BpResult<f64> {
        match &self.inner {
            InnerScalingLpVectorDissimilarity::Trained { meta_data_ranges } => {
                let (min, max) = meta_data_ranges[idx];
                if max - min < 1e-9 {
                    Ok(0.0)
                } else {
                    Ok((value - min) / (max - min) * self.weights[idx])
                }
            }
            _ => Err("not trained".into()),
        }
    }
}

impl Default for ScalingLpVectorDissimilarity {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainable for ScalingLpVectorDissimilarity {
    fn train<'i>(&mut self, _training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        match &self.inner {
            InnerScalingLpVectorDissimilarity::Untrained => {
                let mut meta_data_ranges = [(f64::INFINITY, f64::NEG_INFINITY); META_DATA_COUNT];
                for immo in _training_data {
                    let meta_data_array = immo.meta_data_array();
                    for (i, meta_datum_option) in meta_data_array.iter().enumerate() {
                        if let Some(meta_datum) = meta_datum_option {
                            meta_data_ranges[i].0 = meta_data_ranges[i].0.min(*meta_datum);
                            meta_data_ranges[i].1 = meta_data_ranges[i].1.max(*meta_datum);
                        }
                    }
                }

                for (idx, (min, max)) in meta_data_ranges.iter_mut().enumerate() {
                    if !min.is_finite() || !max.is_finite() {
                        log::warn!("Meta datum {} had no values to train on. Not scaling.", idx);
                        *min = 0.0;
                        *max = 1.0;
                    }
                }

                self.inner = InnerScalingLpVectorDissimilarity::Trained { meta_data_ranges };
                Ok(())
            }
            _ => Err("was already trained".into()),
        }
    }
}

impl Dissimilarity for ScalingLpVectorDissimilarity {
    fn dissimilarity(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        immo1
            .meta_data_array()
            .iter()
            .zip(immo2.meta_data_array().iter())
            .enumerate()
            .map(|(i, options)| match options {
                (Some(value1), Some(value2)) => (self.scale(*value1, i).expect("was not trained")
                    - self.scale(*value2, i).expect("was not trained"))
                .abs()
                .powf(self.exponent),
                _ => 0.0,
            })
            .sum::<f64>()
            .powf(1.0 / self.exponent)
    }
}

/// a trainable version of [ManhattanVectorDissimilarity] which normalizes all values to their z-score
/// values after nomalization are `(value - mean) / std_dev`
/// z-score of different attributes can also be weighted
#[derive(Debug, Clone)]
pub struct NormalizingLpVectorDissimilarity {
    inner: InnerNormalizingLpVectorDissimilarity,
    weights: [f64; META_DATA_COUNT],
    exponent: f64,
}

#[derive(Debug, Clone, PartialEq)]
enum InnerNormalizingLpVectorDissimilarity {
    Untrained,
    Trained {
        means: [f64; META_DATA_COUNT],
        std_deviations: [f64; META_DATA_COUNT],
    },
}

impl NormalizingLpVectorDissimilarity {
    /// Creates a new dissimilarity with all weights equal to 1.0
    pub fn new() -> Self {
        Self::with_weights([1.0; META_DATA_COUNT])
    }

    /// Creates a new dissimilarity with given weights
    pub fn with_weights(weights: [f64; META_DATA_COUNT]) -> Self {
        Self::with_weights_and_exponent(weights, 1.0)
    }

    /// Creates a new dissimilarity with given weights and exponent `p`
    pub fn with_weights_and_exponent(weights: [f64; META_DATA_COUNT], exponent: f64) -> Self {
        assert!(
            (0.1..=10.0).contains(&exponent),
            "exponent has to be between 0.1 and 10 to avoid accidental infinity as result"
        );
        Self {
            inner: InnerNormalizingLpVectorDissimilarity::Untrained,
            weights,
            exponent,
        }
    }

    fn normalize(&self, value: f64, idx: usize) -> BpResult<f64> {
        match &self.inner {
            InnerNormalizingLpVectorDissimilarity::Trained {
                means,
                std_deviations,
            } => {
                if std_deviations[idx] > 1e-9 {
                    Ok((value - means[idx]) / std_deviations[idx] * self.weights[idx])
                } else {
                    Ok(0.0)
                }
            }
            _ => Err("was not trained".into()),
        }
    }
}

impl Default for NormalizingLpVectorDissimilarity {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainable for NormalizingLpVectorDissimilarity {
    fn train<'i>(&mut self, training_data: impl IntoIterator<Item = &'i Immo>) -> BpResult<()> {
        match &self.inner {
            InnerNormalizingLpVectorDissimilarity::Untrained => {
                let training_data: Vec<_> = training_data.into_iter().collect();
                let mut means = [0.0; META_DATA_COUNT];
                for (i, mean) in means.iter_mut().enumerate() {
                    let values: Vec<_> = training_data
                        .iter()
                        .filter_map(|immo| immo.meta_data_array()[i])
                        .collect();
                    if values.is_empty() {
                        return Err("Some attributes had no values to train on.".into());
                    }

                    *mean = values.iter().sum::<f64>() / values.len() as f64;
                }

                let mut std_deviations = [1.0; META_DATA_COUNT];
                for i in 0..META_DATA_COUNT {
                    let values: Vec<_> = training_data
                        .iter()
                        .filter_map(|immo| immo.meta_data_array()[i])
                        .map(|value| (means[i] - value) * (means[i] - value))
                        .collect();
                    let variance = values.iter().sum::<f64>() / values.len() as f64;
                    std_deviations[i] = variance.sqrt();
                }

                self.inner = InnerNormalizingLpVectorDissimilarity::Trained {
                    std_deviations,
                    means,
                };

                Ok(())
            }
            _ => Err("was already trained".into()),
        }
    }
}

impl Dissimilarity for NormalizingLpVectorDissimilarity {
    fn dissimilarity(&self, immo1: &Immo, immo2: &Immo) -> f64 {
        immo1
            .meta_data_array()
            .iter()
            .zip(immo2.meta_data_array().iter())
            .enumerate()
            .map(|(i, options)| match options {
                (Some(value1), Some(value2)) => {
                    (self.normalize(*value1, i).expect("was not trained")
                        - self.normalize(*value2, i).expect("was not trained"))
                    .abs()
                    .powf(self.exponent)
                }
                _ => 0.0,
            })
            .sum::<f64>()
            .powf(1.0 / self.exponent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use chrono::NaiveDate;
    use common::immo::{MicroLocationScore, Zustand};
    use proptest::prelude::*;
    use std::convert::TryInto;
    use test_helpers::*;

    fn get_example_immos() -> Vec<Immo> {
        vec![
            Immo {
                baujahr: Some(1950),
                wohnflaeche: Some(100.0),
                zustand: Some(Zustand::Mittel),
                ausstattung: Some(1),
                grundstuecksgroesse: Some(100.0),
                anzahl_stellplaetze: Some(10),
                wertermittlungsstichtag: Some(NaiveDate::from_ymd(2014, 1, 1)),
                plane_location: Some((1.0, 2.0)),
                centrality: Some(10),
                regiotyp: Some(10),
                restnutzungsdauer: Some(5.0),
                micro_location_scores: Some(MicroLocationScore {
                    all: 10.0,
                    education_and_work: 10.0,
                    leisure: 10.0,
                    public_transport: 10.0,
                    shopping: 10.0,
                }),
                ..Immo::default()
            },
            Immo {
                baujahr: Some(1960),
                wohnflaeche: Some(110.0),
                zustand: Some(Zustand::Gut),
                ausstattung: Some(2),
                grundstuecksgroesse: Some(120.0),
                anzahl_stellplaetze: Some(8),
                wertermittlungsstichtag: Some(NaiveDate::from_ymd(2015, 1, 1)),
                plane_location: Some((2.0, 3.0)),
                centrality: Some(12),
                regiotyp: Some(13),
                restnutzungsdauer: Some(10.0),
                micro_location_scores: Some(MicroLocationScore {
                    all: 11.0,
                    education_and_work: 11.0,
                    leisure: 11.0,
                    public_transport: 11.0,
                    shopping: 11.0,
                }),
                ..Immo::default()
            },
        ]
    }

    macro_rules! generate_overlapping_manhattan_vector_dissimilarity_tests {
        ($name: ident, $dissimilarity_type: ty) => {
            mod $name {
                use super::*;

                #[test]
                fn cant_train_twice() {
                    let immos = get_example_immos();
                    let mut dissimilarity = <$dissimilarity_type>::new();
                    dissimilarity.train(&mut immos.iter()).unwrap();
                    assert!(dissimilarity.train(&mut immos.iter()).is_err());
                }

                #[test]
                #[should_panic]
                fn panics_if_not_trained() {
                    let immos = get_example_immos();
                    let dissimilarity = <$dissimilarity_type>::new();
                    dissimilarity.dissimilarity(&immos[0], &immos[1]);
                }

                proptest! {
                    #[test]
                    fn is_neutral(
                        immos in prop::collection::vec(full_immo(), 2..128),
                        weights in proptest::collection::vec(1e-3..1e3, META_DATA_COUNT..=META_DATA_COUNT),
                        exponent in 0.1..10.0,
                    ) {
                        let mut dissimilarity =
                            <$dissimilarity_type>::with_weights_and_exponent(weights.try_into().unwrap(), exponent);
                        let res = dissimilarity.train(&mut immos.iter());
                        if res.is_ok() { // if every field had at least two different values
                            for immo in immos {
                                prop_assert!(dissimilarity.dissimilarity(&immo, &immo).abs() < 1e-9);
                            }
                        }
                    }
                }
            }
        }
    }

    generate_overlapping_manhattan_vector_dissimilarity_tests!(
        scaling_lp_vector_dissimilarity,
        ScalingLpVectorDissimilarity
    );
    generate_overlapping_manhattan_vector_dissimilarity_tests!(
        normalizing_manhattan_vector_dissimilarity,
        NormalizingLpVectorDissimilarity
    );

    proptest! {
        #[test]
        fn scaling_lp_vector_dissimilarity_gives_correct_output(
            weights in proptest::collection::vec(1e-3..1e3, META_DATA_COUNT..=META_DATA_COUNT),
            exponent in 0.1..10.0,
        ) {
            let immos = get_example_immos();
            let mut dissimilarity =
                ScalingLpVectorDissimilarity::with_weights_and_exponent(weights.clone().try_into().unwrap(), exponent);
            dissimilarity.train(&mut immos.iter()).unwrap();
            assert_approx_eq!(
                dissimilarity.dissimilarity(&immos[0], &immos[1]),
                weights
                    .iter()
                    .map(|value| value.powf(exponent))
                    .sum::<f64>()
                    .powf(1.0 / exponent)
            );
        }
    }

    #[test]
    fn scaling_lp_vector_dissimilarity_is_okay_with_only_one_value() {
        let mut immos = get_example_immos();
        immos[0].baujahr = Some(2000);
        immos[1].baujahr = Some(2000);
        let mut dissimilarity =
            ScalingLpVectorDissimilarity::with_weights([22.0 / 7.0; META_DATA_COUNT]);
        assert!(dissimilarity.train(&mut immos.iter()).is_ok());
    }

    #[test]
    fn from_config() {
        let config = ScalingLpVectorDissimilarityConfig {
            weights: [1.0; META_DATA_COUNT],
            exponent: 1.0,
        };

        assert_eq!(
            ScalingLpVectorDissimilarity::default(),
            ScalingLpVectorDissimilarity::from_config(&config).unwrap()
        );
    }
}
