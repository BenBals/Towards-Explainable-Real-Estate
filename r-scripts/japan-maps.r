setwd("~/Studium/BP/git-python/r-scripts")
source("database.r")
source("helpers.r")

limit = 100000000
hpi_red <- "#B30047"
hpi_blue <- '#007396'
dir.create("export")

logger_fun <- function(x) {
  log(x * 100)
}
logger_inv <- function(x) {
  exp(x) / 100
}
logger <-
  trans_new(name = "logging",
            transform = logger_fun,
            inverse = logger_inv)

mybreaks <- c(0.1, 0.2, 0.3, 0.4, 0.5)

library(dplyr)
library(ggplot2)
library(rgdal)
library(rgeos)
library(purrr)
library(stringr)
library(sf)
library(spatstat)
library(stars)

shapefile <-
  readOGR("japan_shapefiles/jpn_admbnda_adm1_2019.shp")
shapefile$prefecture <-
  (shapefile$ADM1_EN %>% map_chr(function(unclean) {
    clean <- str_trim(gsub("\\W+", "", unclean), side = "both")
    return(clean)
  }))
shapefile_sf <-
  st_read("japan_shapefiles/jpn_admbnda_adm1_2019.shp") %>% st_transform(2459)


clean_data <-
  load_data(
    attributes = c(
      "predictions",
      "marktwert",
      "kurzgutachten.objektangabenWohnflaeche",
      "plane_location",
      "prefecture"
    ),
    limit = limit
  )
unclean_data <-
  load_unclean_data(
    attributes = c(
      "predictions",
      "marktwert",
      "kurzgutachten.objektangabenWohnflaeche",
      "plane_location",
      "prefecture"
    ),
    limit = limit
  )

clean_data_pe_w_prefecture <-
  clean_data %>% 
  filter(!is.na(prefecture)) %>% 
  mutate(prediction = `predictions.dully-split201703-japan`) %>% 
  mutate(pe = (marktwert - prediction) / marktwert) %>% 
  mutate(ape = abs(pe)) %>% filter(ape > 0.5) %>% 
  select(`_id`, pe, prefecture, lng, lat)

unclean_data_pe_w_prefecture <-
  unclean_data %>% filter(!is.na(prefecture)) %>% 
  mutate(prediction = `predictions.dully-split201703-japan`) %>% 
  mutate(pe = (marktwert - prediction) / marktwert) %>% 
  mutate(abs = abs(pe)) %>% 
  select(`_id`, pe, prefecture, lng, lat)


clean_outliers <-
  clean_data_pe_w_prefecture %>% select(lat, lng) %>% st_as_sf(coords = c("lng", "lat"))
st_crs(clean_outliers) <- 2459
unclean_points <- 
  unclean_data_pe_w_prefecture %>% select(lat, lng) %>% st_as_sf(coords = c("lng", "lat"))
st_crs(unclean_points) <- 2459

ggplot() +
  geom_sf(data = shapefile_sf) +
  geom_sf(data = clean_outliers, col = hpi_blue, size = 0.15) +
  geom_sf(data = unclean_points, col = hpi_red, size = 0.15) +
  theme_void()

ggsave("export/outliers_points.png", scale = 10, limitsize = FALSE)

clean_outliers_per_prefecture <- clean_data_pe_w_prefecture %>% count(prefecture) %>% rename(num_clean_outliers = n)
unclean_data_per_prefecture <- unclean_data_pe_w_prefecture %>% count(prefecture) %>% rename(num_unclean_data = n)
clean_outliers_per_unclean <- merge(clean_outliers_per_prefecture, unclean_data_per_prefecture, by="prefecture") %>% summarise(prefecture, clean_outliers_per_unclean = num_clean_outliers / num_unclean_data) 

merged <- merge(shapefile, clean_outliers_per_prefecture, by= "prefecture", all.x = TRUE)
merged <- merge(merged, unclean_data_per_prefecture, by= "prefecture", all.x = TRUE)
merged <- merge(merged, clean_outliers_per_unclean, by= "prefecture", all.x = TRUE)

tidy_merged <- tidy(merged)
merged$id <- row.names(merged)
tidy_merged <- left_join(tidy_merged, merged@data)

make_plot <- function(data_column_name, name) {
  plotti <- ggplot(tidy_merged,
                   aes(
                     x = long,
                     y = lat,
                     group = group,
                     fill = .data[[data_column_name]]
                   )) +
    geom_polygon(color = "black", size = 0.1) +
    coord_equal() +
    theme_void() +
    scale_fill_gradient(
      name = name,
      trans = logger,
      high = hpi_red,
      low = hpi_blue,
      breaks = mybreaks,
      labels = mybreaks
    ) +
    labs(title = glue("Per prefecture on {name}")) +
    theme(plot.title = element_text(margin = margin(t = 40, b = -40)),
          legend.position = c(0.75, 0.55),
    )
  return(plotti)
}

plot <- make_plot("num_clean_outliers", "clean_outliers")
plot2 <- make_plot("num_unclean_data", "unclean data")
plot3 <- make_plot("clean_outliers_per_unclean", "clean outliers per unclean")
show(plot)
show(plot2)
show(plot3)
