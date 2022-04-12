let types = ./types.dhall

in    types.DullyConfig::{
      , dissimilarity = types.DissimilarityConfig.Dully
      , neighbor_selection_strategy =
          types.NeighborSelectionStrategy.Radius 10000.0
      }
    : types.DullyConfig.Type
