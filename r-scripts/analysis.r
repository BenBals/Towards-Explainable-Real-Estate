# setwd("~/Studium/BP/git-python/r-scripts/")
# setwd("~/code/bp/db_scripts/r-scripts")
setwd("~/Documents/BP/db_scripts/r-scripts")

# optional, if saved data should be loaded
load("./with_data.RData")

source("database.r")
source("helpers.r")

library(dplyr)
library(ggplot2)
library(MLmetrics)
library(cowplot)
library(gridExtra)
library(reshape2)
library(ggthemes)
library(glue)

dir.create("export")

data <- load_data(attributes=c("predictions", "marktwert", "kurzgutachten.objektangabenWohnflaeche", 'test-set'), limit = 1000000)
#data_copy <- data.frame(data)

# only do for testing
#data <- sample_n(data_copy, 1000)

random_experiments = c(
  'predictions.cbr-ea-m-infty-random-train-test-split--clean',
  'predictions.cbr-ea-m-infty-random-train-test-split--unclean',
  'predictions.cbr-ea-m-10-random-train-test-split--clean',
  'predictions.cbr-ea-m-10-random-train-test-split--unclean',
  'predictions.cbr-lbs-random-train-test-split-clean',
  'predictions.cbr-lbs-random-train-test-split-unclean',
  
  'predictions.dnn-series3-tabnet-clean-random-split-cbr-lbs',
  'predictions.dnn-series3-tabnet-unclean-random-split-cbr-lbs',
  'predictions.dnn-series3-kaggle-baseline-clean-random-split-cbr-lbs',
  'predictions.dnn-series3-kaggle-baseline-unclean-random-split-cbr-lbs',
  'predictions.dnn-series3-kaggle-house-prices-clean-random-split-cbr-lbs',
  'predictions.dnn-series3-kaggle-house-prices-unclean-random-split-cbr-lbs'
)

date_experiments = c("predictions.cbr-ea-m-infty-clean", 
                     "predictions.cbr-ea-m-infty-unclean", 
                     "predictions.cbr-ea-m-10-clean", 
                     "predictions.cbr-ea-m-10-unclean",
                     
                     "predictions.dnn-series5-tabnet-clean-random-val-split-cbr-lbs",
                     "predictions.dnn-series5-tabnet-unclean-random-val-split-cbr-lbs",
                     "predictions.dnn-series3-kaggle-baseline-clean-random-val-split-cbr-lbs",
                     "predictions.dnn-series3-kaggle-baseline-unclean-random-val-split-cbr-lbs",
                     "predictions.dnn-series3-kaggle-house-prices-clean-random-val-split-cbr-lbs",
                     "predictions.dnn-series3-kaggle-house-prices-unclean-random-val-split-cbr-lbs",
                     "predictions.dnn-series5-tabnet-clean-random-val-split-no-cbr",
                     "predictions.dnn-series5-tabnet-unclean-random-val-split-no-cbr",
                     "predictions.dnn-series5-kaggle-baseline-clean-random-val-split-no-cbr",
                     "predictions.dnn-series5-kaggle-baseline-unclean-random-val-split-no-cbr",
                     "predictions.dnn-series5-kaggle-house-prices-clean-random-val-split-no-cbr",
                     "predictions.dnn-series5-kaggle-house-prices-unclean-random-val-split-no-cbr",
                     'predictions.dnn-series6-tabnet-long-clean-random-val-split-cbr-lbs',
                     'predictions.dnn-series6-tabnet-long-unclean-random-val-split-cbr-lbs'
                     )

dully_experiments <- c(
  "predictions.dully-split201703-cleanv3",
  "predictions.dully-split201703-uncleanv3"
)
# TODO: do something with dullys

experiment_names <- c(random_experiments, date_experiments)

# testing only
#for (exp in experiment_names) {
#  for(i in 6:10) {
#    data[[glue('{exp}.{i}')]] = NULL
#  }
#}

num_runs_per_experiment <- 10

run_id_to_experiment_id <- function (run_id) {
  for (experiment in experiment_names) {
    if (grepl(experiment, run_id, fixed = TRUE)) {
      return(experiment)
    }
  }
  return("")
}
bool_to_number <- function(b) {
  if (!is.na(b)) {
    return(1)
  } else {
    return(NA)
  }
}

last = ''
for (exp in random_experiments) {
  for (run in 1:num_runs_per_experiment) {
    data$tmp <- data[[glue('{exp}-{run}')]]
    data$bascht <- data[[glue('test-set.random-{run}')]]
    last = glue('{exp}-{run}')
    data <- data %>% rowwise() %>% mutate(newtmp = (tmp * bool_to_number(bascht)))
    data[[glue('{exp}-{run}')]] <- data$newtmp
  }
}
for (exp in date_experiments) {
  for (run in 1:num_runs_per_experiment) {
    data$tmp <- data[[glue('{exp}-{run}')]]
    data$bascht <- data[[glue('test-set.date-20170301')]]
    data <- data %>% rowwise() %>% mutate(newtmp = tmp * bool_to_number(bascht))
    data[[glue('{exp}-{run}')]] <- data$newtmp
  }
}
for (exp in dully_experiments) {
  data$tmp <- data[[exp]]
  data$bascht <- data[[glue('test-set.date-20170301')]]
  data <- data %>% rowwise() %>% mutate(newtmp = tmp * bool_to_number(bascht))
  data[[exp]] <- data$newtmp
}
data$bascht <- NULL
data$tmp <- NULL
data$newtmp <- NULL

cols <- colnames(data)
prediction_cols <- cols[grepl("predictions", cols)]

# multiply dnn predictions by wohnflaeche
dnn_cols <- prediction_cols[grepl("dnn", prediction_cols)]
dnn_cols <- dnn_cols[!grepl("fixed", dnn_cols)]
for (col in dnn_cols) {
  data[[glue("fixed-{col}")]] <- data[["kurzgutachten.objektangabenWohnflaeche"]] * data[[col]]
  data[[col]] <- NULL
}

# load the fixed- cols into prediction_cols
cols <- colnames(data)
prediction_cols <- cols[grepl("predictions", cols)]
prediction_cols <- prediction_cols[!grepl("pe_", prediction_cols)]

pe_cols = c()
for (col in prediction_cols)  {
  pe_name <- glue("pe_{col}")
  data[[pe_name]] <- (data$marktwert - data[[col]]) / data$marktwert
  pe_cols <- append(pe_cols, pe_name)
}


run_summary_df <- analysis_by_run()
mape_summary_df <- run_summary_df %>% rowwise() %>% mutate(exp=run_id_to_experiment_id(name)) %>% group_by(exp) %>% summarise(mean(MAPE), var(MAPE), sd(MAPE))
write.csv2(run_summary_df, "export/run_summaries.csv")
write.csv2(mape_summary_df, "export/mean_of_mapes.csv")

# VIOLIN PLOTS
data_melted <- data %>% select(pe_cols) %>% melt(value.name = "prediction", variable.name = "run") %>% fortify()

pe_cols_for_experiment <- function(exp) {
   experiment_pe_cols <- pe_cols[grepl(exp, pe_cols)]
}

base_plot_per_experiment <- function(experiment) {
  experiment_pe_cols <- pe_cols_for_experiment(experiment)
  
  melted <- data %>% select(experiment_pe_cols) %>%
    melt(value.name = "pe", variable.name = "run") %>%
    fortify() %>%
    rowwise() %>%
    mutate(run_readable = extract_run_number(run)) %>%
    filter(!is.na(pe))
  
  without_outliers <- melted %>% filter(pe > -1 & pe < 1)
  
  return(without_outliers %>%
    ggplot(aes(x=factor(run_readable), y=pe)) +
      labs(
      title = "Percentage error over 10 runs",
      subtitle = experiment,
      caption = glue("{round(100*(1-nrow(without_outliers)/nrow(melted)), 1)}% of data with pe <-100% or >100% were removed to make the diagram readable"),
      x="run",
      y="percentage error", 
    ) + theme_economist() + scale_fill_economist() +
    scale_y_continuous(labels = scales::percent) +
    theme(axis.text.y = element_text(hjust = 0.5)))
}

save.image("after_plot_setup.RData")
if (!exists("data")) {
  load("after_plot_setup.RData")
}

for (exp in experiment_names) {
  plot <- base_plot_per_experiment(exp)
  ggsave(glue("export/violinplot_{exp}.png"), plot = plot + geom_violin(), height=5)
  ggsave(glue("export/boxplot_{exp}.png"), plot = plot + geom_boxplot(), height=5)
}

experiment_short_names <- function(experiment) {
  exp <- as.character(experiment)
  if (exp=="predictions.cbr-ea-m-infty-clean") { return ("m=∞, clean") }
  if (exp=="predictions.cbr-ea-m-infty-unclean") {return("m=∞, unclean")}
  if (exp=="predictions.cbr-ea-m-10-clean") {return("m=10, clean")}
  if (exp=="predictions.cbr-ea-m-10-unclean") {return("m=10, unclean")}
  if (exp=="predictions.dnn-series3-kaggle-baseline-clean-random-val-split-cbr-lbs") {return("kaggle-baseline, lbs clean")}
  if (exp=="predictions.dnn-series3-kaggle-baseline-unclean-random-val-split-cbr-lbs") {return("kaggle-baseline, lbs unclean")}
  if (exp=="predictions.dnn-series5-tabnet-clean-random-val-split-cbr-lbs") {return("tabnet, lbs clean")}
  if (exp=="predictions.dnn-series5-tabnet-unclean-random-val-split-cbr-lbs") {return("tabnet, lbs unclean")}
  if (exp=="predictions.dnn-series3-kaggle-house-prices-clean-random-val-split-cbr-lbs") {return("kaggle-house-prices, lbs clean")}
  if (exp=="predictions.dnn-series3-kaggle-house-prices-unclean-random-val-split-cbr-lbs") {return("kaggle-house-prices, lbs unclean")}
  
  return("invalid")
}

experiment_numeric_ids <- function(experiment) {
  exp <- as.character(experiment)
  if (exp=="predictions.cbr-ea-m-infty-clean") { return (1) }
  if (exp=="predictions.cbr-ea-m-infty-unclean") {return(2)}
  if (exp=="predictions.cbr-ea-m-10-clean") {return(3)}
  if (exp=="predictions.cbr-ea-m-10-unclean") {return(4)}
  if (exp=="predictions.dnn-series3-kaggle-baseline-clean-random-val-split-cbr-lbs") {return(5)}
  if (exp=="predictions.dnn-series3-kaggle-baseline-unclean-random-val-split-cbr-lbs") {return(7)}
  if (exp=="predictions.dnn-series5-tabnet-clean-random-val-split-cbr-lbs") {return(8)}
  if (exp=="predictions.dnn-series5-tabnet-unclean-random-val-split-cbr-lbs") {return(9)}
  if (exp=="predictions.dnn-series3-kaggle-house-prices-clean-random-val-split-cbr-lbs") {return(10)}
  if (exp=="predictions.dnn-series3-kaggle-house-prices-unclean-random-val-split-cbr-lbs") {return(11)}
  
  return(0)
}


short_name = experiment_short_names

create_aggregated_plots <- function(short_name) {
  gc()
  melted_by_exp <- data %>% select(pe_cols) %>%
    melt(value.name = "pe", variable.name = "run")
  melted_by_exp <- melted_by_exp %>%
    fortify()
  melted_by_exp$exp <- lapply(melted_by_exp$run, function(run) {experiment_numeric_ids(run_id_to_experiment_id(run))})
  melted_by_exp <- melted_by_exp %>% rowwise() %>%
    filter(exp != 0) %>%
    select(exp, pe)
  melted_by_exp$exp <- as.factor(melted_by_exp$exp)
  
  save.image("after_large_melting.RData")
  load("after_large_melting.RData")
  
  gc()
  
  without_outliers_by_exp <- melted_by_exp %>% filter(pe > -1 & pe < 1)
  
  aggregated_base_plot <- without_outliers_by_exp %>%
    ggplot(aes(x=exp, y=pe)) + 
      labs(
      title = "Percentage error over 10 runs",
      caption = glue("{round(100*(1-nrow(without_outliers_by_exp)/nrow(melted_by_exp)), 1)}% of data with pe <-100% or >100% were removed to make the diagram readable"),
      x="run",
      y="percentage error", 
    ) + theme_economist() + scale_fill_economist() +
    scale_y_continuous(labels = scales::percent) +
    theme(axis.text.y = element_text(hjust = 0.5))
  
  ggsave("export/violinplot_all_experiment.png", plot=aggregated_base_plot + geom_violin(), height=5, width=10)
  ggsave("export/boxplot_all_experiment.png", plot=aggregated_base_plot + geom_boxplot(), height=5, width=10)
  
  gc()
  
  without_outliers_by_exp %>%
    ggplot(aes(color=exp, x=pe)) + geom_density() +
      labs(
      title = "Distribution of percentage errors",
      subtitle = "Each config is aggregated over 10 runs",
      caption = glue("{round(100*(1-nrow(without_outliers_by_exp)/nrow(melted_by_exp)), 1)}% of data with pe <-100% or >100% were removed to make the diagram readable"),
      x="percentage error",
      y="density", 
      color = "Config"
    ) + theme_economist() + scale_fill_economist() +
    scale_y_continuous(labels = scales::percent) +
    theme(axis.text.y = element_text(hjust = 0.5))
  ggsave("export/densityplot_all_experiment.png", height=5, width=10)
}
create_aggregated_plots(experiment_short_names)

# Mean sd between the predictions for the same house by the same algorithms between different runs
mean_sd <- function() {
  mean_sd_between_runs <- data.frame(matrix(ncol = 2, nrow = 0))
  colnames(mean_sd_between_runs) <- c('experiment', 'mean_sd_between_runs')
  
  for (exp in experiment_names) {
    sd_between_runs <- data %>%
      select(pe_cols_for_experiment(exp)) %>%
      apply(1, FUN = sd, na.rm = TRUE)
    
    mean_sd_between_runs[nrow(mean_sd_between_runs) + 1,] = c(exp, mean(sd_between_runs, na.rm=TRUE))
  }
  
  write.csv2(mean_sd_between_runs, "export/mean_sd_between_runs.csv")
}
mean_sd()


# For testing: A U-test between a single cbr run and a single dnn run
ea_pes <- data$`pe_predictions.cbr-ea-m-10-clean-1`
dnn_pes <- data$`pe_predictions.dnn-ea-no-cbr-clean-series3-fenrir-1-fixed`

wilcox.test(abs(ea_pes), abs(dnn_pes), alternative = "less")
