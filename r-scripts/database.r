library(mongolite)
library(purrr) # functional programming helpers for R
library(jsonlite)
library(glue)

default_attributes_to_load = c("marktwert", "kurzgutachten.objektangabenWohnflaeche", "predictions", "plane_location")
basic_query_string = '"plane_location.0": { "$gt": -1140843.77, "$lt": 1414596.17 },
      "plane_location.1": { "$gt": -401614.57, "$lt": 2618746.47 },
      "duplicate": { "$ne": true },
      "objektunterart": { "$ne": "invalid" },
      "last_entry_with_bukken_id": true '
clean_data_query_string = '"kurzgutachten.objektangabenBaujahr": { "$gt": 1500, "$lt": 2025 },
      "kurzgutachten.objektangabenWohnflaeche": { "$gt": 20, "$lt": 2000 },
      "marktwert": { "$lte": 300000000 }'
unclean_data_query_string = '"$or": [ {"kurzgutachten.objektangabenBaujahr": { "$not": { "$gt": 1500, "$lt": 2025 } } },
      {"kurzgutachten.objektangabenWohnflaeche": { "$not": { "$gt": 20, "$lt": 40 } }},
      {"marktwert": { "$gt": 300000000 }} ]'

get_mongo_before_date <- function(date_string) {
  s <- paste('"wertermittlungsstichtag": {"$gte": {"$date": "', date_string, '"} }', sep = "")
  return(s)
}

flatten_location <- function(data) {
  if ('plane_location' %in% colnames(data)) {
    data$lng <- mapply(function(a)
      a[[1]], data$plane_location)
    data$lat <- mapply(function(a)
      a[[2]], data$plane_location)
    data <- subset(data, select = -c(plane_location))
  }
  return(data)
}

load_data_with_query <-
  function(limit,
           attributes,
           collection_name,
           query_string) {
    coll <-
      mongo(
        collection_name,
        url = glue(
          "mongodb://{username}:{password}@localhost:27017/japan_lifullhome?authSource=admin"
        )
      )
    
    data <-
      coll$find(query = query_string,
                fields = attribute_list_to_mongo_json(attributes),
                limit = limit)
    data <- jsonlite::flatten(data)
    data <- flatten_location(data)
    return(data)
  }

load_data <-
  function(limit = 10000,
           attributes = default_attributes_to_load,
           collection_name = "japan_sales_reshaped") {
    # define secret in console
    clean_query_string = paste("{", basic_query_string, ",", clean_data_query_string, "}" )
    print(clean_query_string)
    data <- load_data_with_query(limit, attributes, collection_name, clean_query_string)
    return(data)
  }
      
load_unclean_data = 
  function(limit = 10000,
           attributes = default_attributes_to_load,
           collection_name = "japan_sales_reshaped") {
    # define secret in console
    unclean_query_string = paste('{ "$and": [ {', basic_query_string, ",", get_mongo_before_date("2017-03-01T00:00Z"), "} , { ", unclean_data_query_string, "} ] }" )
    print(unclean_query_string)
    data <- load_data_with_query(limit, attributes, collection_name, unclean_query_string)
    return(data)
}

attribute_list_to_mongo_json <- function(attributes_list) {
  str_vec = c()
  for (attr in attributes_list) {
    str_vec = append(str_vec, paste('"', attr, '": true', sep = ""))
  }
  result = paste(str_vec, collapse = ", ")
  result = paste("{", result, "}")
  return(result)
}
