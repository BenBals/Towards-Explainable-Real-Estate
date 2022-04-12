let types = ./types.dhall

let Config = types.Config

let BaBasis = ./ba-basis.dhall

in  BaBasis.default : Config.Type
