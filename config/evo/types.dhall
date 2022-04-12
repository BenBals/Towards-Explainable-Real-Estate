let FitnessSharingConfig =
      { Type =
          { -- If two indivudal's distance is below this threshold they will be punished in fitness sharing (sigma in Martin's paper)
            distance_threshold : Double
          , -- Configures how strong the effect of fitness sharing should be (alpha in Martin's paper)
            exponent : Double
          }
      , default = { distance_threshold = 0.3, exponent = 2.0 }
      }

let Config =
      { Type =
          { -- how much of the training data should be used for validation?
            train_test_split_ratio : Double
          , -- how many iterations for the evo
            num_generations : Natural
          , --
            population_size : Natural
          , -- how many immos do you predict in one iteration
            batch_size : Natural
          , -- how many individuals are used to produce one offspring
            individuals_per_parent : Natural
          , -- how likely does an entry in an individual get mutated? A value of x / META_DATA_COUNT results in expectedly x mutations
            mutation_rate : Double
          , -- percentage of individuals from the offspring which do replace the old population
            replace_ratio : Double
          , -- standard deviation of value change in mutation
            std_deviation : Double
          , -- What percentage of individuals should be selected as parents?
            selection_ratio : Double
          , -- How like will an enum switch its variant?
            type_switch_probability : Double
          , -- After how many unsuccessful generations should the evo restart?
            multistart_threshold : Natural
          , -- fitness sharing comment
            fitness_sharing : Optional FitnessSharingConfig.Type
          , -- should the evo use a mutation strategy of annealing?
            simulated_annealing : Bool
          , -- BA only (don't merge this)
            strict_radius_limit : Optional Double
          , ignore_filters : Bool
          , k_most_similar_limit : Optional Natural
          , fix_exponent : Optional Double
          , -- How big should be the step for the local search. Should be in (0, 1]
            local_search_factor : Double
          , -- Should the best weight as stated in the code be used as the initial seed?
            best_result_as_seed : Bool
        , disable_normalization : Bool
          }
      , default =
        { train_test_split_ratio = 0.8
        , num_generations = 200
        , population_size = 20
        , batch_size = 10000
        , individuals_per_parent = 5
        , mutation_rate = 0.2
        , replace_ratio = 0.85
        , std_deviation = 0.2
        , selection_ratio = 1.0
        , type_switch_probability = 0.05
        , multistart_threshold = 10
        , fitness_sharing = None FitnessSharingConfig.Type
        , simulated_annealing = False
        , strict_radius_limit = None Double
        , ignore_filters = False
        , k_most_similar_limit = None Natural
        , fix_exponent = None Double
        , local_search_factor = 1.0
        , best_result_as_seed = False
        , disable_normalization = False
        }
      }

in  { Config, FitnessSharingConfig, }
