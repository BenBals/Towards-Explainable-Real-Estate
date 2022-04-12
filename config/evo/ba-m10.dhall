let types = ./types.dhall

let Config = types.Config

let BaBasis = ./ba-basis.dhall

in  BaBasis::{ k_most_similar_limit = Some 10 } : Config.Type
