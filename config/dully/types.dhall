let ScalingLpVectorDissimilarityConfig =
      { Type = { weights : List Double, exponent : Double }
      , default =
        { weights =
          [ 1.0
          , 1.0
          , 1.0
          , 1.0
          , 1.0
          , 1.0
          , 1.0
          , 1.0
          , 1.0
          , 1.0
          , 1.0
          , 1.0
          , 1.0
          , 1.0
          ]
        , exponent = 1.0
        }
      }

let ExpertDissimilarityConfigType =
      { cutoff_similarity : Optional Double
      , similarity_part_1_weight : Double
      , wohnflaeche_factor_category_1_2 : Double
      , wohnflaeche_factor_category_3 : Double
      , wohnflaeche_weight : Double
      , plane_distance_factor : Double
      , plane_distance_weight : Double
      , baujahr_factor : Double
      , baujahr_weight : Double
      , grundstuecksgroesse_factor : Double
      , grundstuecksgroesse_weight : Double
      , anzahl_stellplaetze_factor : Double
      , anzahl_stellplaetze_weight : Double
      , anzahl_zimmer_factor : Double
      , anzahl_zimmer_weight : Double
      , combined_location_score_factor : Double
      , combined_location_score_weight : Double
      , verwendung_weight : Double
      , keller_weight : Double
      }

let ExpertDissimilarityConfig =
      { Type = ExpertDissimilarityConfigType
      , default =
            { cutoff_similarity = None Double
            , similarity_part_1_weight = 0.9
            , wohnflaeche_factor_category_1_2 = 0.4
            , wohnflaeche_factor_category_3 = 2.0
            , wohnflaeche_weight = 1.0
            , plane_distance_factor = 2.5
            , plane_distance_weight = 1.0
            , baujahr_factor = 2.0
            , baujahr_weight = 1.0
            , grundstuecksgroesse_factor = 0.2
            , grundstuecksgroesse_weight = 1.0
            , anzahl_stellplaetze_factor = 25.0
            , anzahl_stellplaetze_weight = 1.0
            , anzahl_zimmer_factor = 50.0
            , anzahl_zimmer_weight = 1.0
            , combined_location_score_factor = 10.0
            , combined_location_score_weight = 1.0
            , verwendung_weight = 1.0
            , keller_weight = 1.0
            }
          : ExpertDissimilarityConfigType
      }

let DissimilarityConfig =
      < Expert : ExpertDissimilarityConfig.Type
      | ScalingLpVector : ScalingLpVectorDissimilarityConfig.Type
      | UnweightedScalingLpVector : Double
      | Cosine
      | Dully
      >

let NeighborSelectionStrategy = < Radius : Double | KNearest : Natural >

let FilterStrategy = < Category : Natural | Continuous : Double >

let FilterConfigType =
      { difference_regiotyp : Optional Natural
      , difference_baujahr : Optional FilterStrategy
      , difference_grundstuecksgroesse : Optional FilterStrategy
      , difference_wohnflaeche : Optional FilterStrategy
      , difference_objektunterart : Optional Natural
      , difference_wertermittlungsstichtag_days : Optional Natural
      , difference_spatial_distance_kilometer : Optional Natural
      , difference_zustand : Optional Natural
      , difference_ausstattungsnote : Optional Natural
      , difference_balcony_area : Optional Double
      , difference_urbanity_score : Optional Natural
      , difference_convenience_store_distance : Optional Double
      , difference_distance_elem_school : Optional Double
      , difference_distance_jun_highschool : Optional Double
      , difference_distance_parking : Optional Double
      , difference_walking_distance : Optional Double
      , difference_floor : Optional Double
      }

let FilterConfig =
      { Type = FilterConfigType
      , default =
            { difference_regiotyp = None Natural
            , difference_baujahr = None FilterStrategy
            , difference_grundstuecksgroesse = None FilterStrategy
            , difference_wohnflaeche = None FilterStrategy
            , difference_objektunterart = None Natural
            , difference_wertermittlungsstichtag_days = None Natural
            , difference_spatial_distance_kilometer = None Natural
            , difference_zustand = None Natural
            , difference_ausstattungsnote = None Natural
            , difference_balcony_area = None Double
            , difference_urbanity_score = None Natural
            , difference_convenience_store_distance = None Double
            , difference_distance_elem_school = None Double
            , difference_distance_jun_highschool = None Double
            , difference_distance_parking = None Double
            , difference_walking_distance = None Double
            , difference_floor = None Double
            }
          : FilterConfigType
      }

let KMostSimilarOnlyPerObjektunterartCategory =
      { Type =
          { category_1 : Natural, category_2 : Natural, category_3 : Natural }
      , default = { category_1 = 5, category_2 = 5, category_3 = 1 }
      }

let NeighborsForAverageStrategy =
      < All
      | KMostSimilarOnly : Natural
      | KMostSimilarOnlyPerCategory :
          KMostSimilarOnlyPerObjektunterartCategory.Type
      >

let DullyConfigType =
      { neighbor_selection_strategy : NeighborSelectionStrategy
      , neighbors_for_average_strategy : NeighborsForAverageStrategy
      , dissimilarity : DissimilarityConfig
      , filters : Optional FilterConfig.Type
      }

let DullyConfig =
      { Type = DullyConfigType
      , default =
            { neighbor_selection_strategy =
                NeighborSelectionStrategy.Radius 10000.0
            , neighbors_for_average_strategy = NeighborsForAverageStrategy.All
            , dissimilarity =
                DissimilarityConfig.ScalingLpVector
                  ScalingLpVectorDissimilarityConfig.default
            , filters = None FilterConfig.Type
            }
          : DullyConfigType
      }

in  { DissimilarityConfig
    , ScalingLpVectorDissimilarityConfig
    , DullyConfig
    , ExpertDissimilarityConfig
    , NeighborSelectionStrategy
    , FilterStrategy
    , FilterConfig
    , NeighborsForAverageStrategy
    , KMostSimilarOnlyPerObjektunterartCategory
    }
