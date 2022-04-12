let types = ./types.dhall

let Config = types.Config

in  { Type = Config.Type
    , default = Config::{
      , num_generations = 200
      , population_size = 20
      , batch_size = 10000
      , multistart_threshold = 10
      , mutation_rate = 0.2
      , strict_radius_limit = Some 40.0
      }
    , disable_normalization = False
    }
