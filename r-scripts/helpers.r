library(dplyr)
library(ggplot2)

extract_run_number <- function(name) {
  name <- as.character(name)
  last <- substr(name, nchar(name), nchar(name))
  if (last == '0') {
    return(substr(name, nchar(name)-1, nchar(name)))
  } else {
    return(last)
  }
}

pe_by_attribute <- function(attribute) {
  data$pred <- data[[attribute]]
  pe_df <- data %>% rowwise() %>% mutate(pe := (marktwert - pred) / marktwert ) %>% select(pe)
  data$pred <- NULL
  
  return (pe_df)
}

boxplot_pe_by_attribute <- function(df)  {
  plot <- ggplot(df, aes(y = pe) ) + geom_boxplot() + coord_cartesian(ylim = quantile(df$pe, c(0.1, 0.9)))
  #show(plot)
  return(plot)
}

mape_by_attribute <- function(attribute) {
  pe_df <- pe_by_attribute(attribute)
  ape_df <- pe_df %>% rowwise() %>% mutate(ape = abs(pe)) 
  return(mean(ape_df$ape, na.rm=TRUE))
}

variance_by_attribute <- function(attribute) {
  pe_df <- pe_by_attribute(attribute)
  return(var(pe_df$pe, na.rm=TRUE))
}

analysis_by_run <- function() {
  col_names <-c("name", "MAPE", "variance", "std dev")
  df <- data.frame(matrix(ncol = length(col_names), nrow = length(prediction_cols)))
  colnames(df) <- col_names
  i <- 1
  for (col in prediction_cols) {
    pes <- (data$marktwert - data[[col]]) / data$marktwert
    new_row <- list(col, 
                    mean(abs(pes), na.rm=TRUE),
                    var(pes, na.rm=TRUE),
                    sd(pes, na.rm=TRUE)
    )
    df[i,] <- new_row
    i <- i + 1
  }  
  return(df)
}

group_by_experiment <- function(df, experiment_name) {
  res <- df[grepl(experiment_name, df$name), ]
  return(res)
}
