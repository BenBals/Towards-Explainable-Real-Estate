let types = ./types.dhall

in    types.DullyConfig::{
      , dissimilarity = types.DissimilarityConfig.Cosine
      , neighbor_selection_strategy =
          types.NeighborSelectionStrategy.Radius 10000.0
      , neighbors_for_average_strategy =
          types.NeighborsForAverageStrategy.KMostSimilarOnly 1
      }
    : types.DullyConfig.Type
