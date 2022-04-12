let types = ./types.dhall

in    types.DullyConfig::{
      , dissimilarity = types.DissimilarityConfig.UnweightedScalingLpVector 2.0
      , neighbor_selection_strategy =
          types.NeighborSelectionStrategy.Radius 10000.0
      , neighbors_for_average_strategy =
          types.NeighborsForAverageStrategy.KMostSimilarOnly 50
      }
    : types.DullyConfig.Type
