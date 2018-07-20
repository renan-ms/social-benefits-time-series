# Core Tidyverse
library(tidyverse)
library(glue)
library(forcats)

# Time Series
library(timetk)
library(tidyquant)
library(tibbletime)

# Visualization
library(cowplot)

# Preprocessing
library(recipes)

# Sampling / Accuracy
library(rsample)
library(yardstick) 

# Modeling
library(keras)

setwd("~/Mestrado/Disciplinas/MDM/paper")

# Load social benefits data
benPrev <- read.csv('BenPrev.csv', sep = ';', dec = ',')

# Transform data into a ts class
benPrev_ts <- ts(benPrev$benPrevTotal, frequency=12, start=c(1997,1))


# Transform data into a tiny class
social_benefits_expenditure <- benPrev_ts %>%
  tk_tbl() %>%
  mutate(index = as_date(index)) %>%
  as_tbl_time(index = index)

social_benefits_expenditure

# Set backtesting strategy (cross validation - splits)
periods_train <- 12 * 10
periods_test  <- 12 * 1
skip_span     <- 11 * 1

# seed to allow reproducibility
set.seed(772018)

rolling_origin_resamples <- rolling_origin(
  social_benefits_expenditure,
  initial    = periods_train,
  assess     = periods_test,
  cumulative = FALSE,
  skip       = skip_span
)

rolling_origin_resamples


# # Plotting function for a single split
# plot_split <- function(split, expand_y_axis = TRUE, alpha = 1, size = 1, base_size = 14) {
# 
#   # Manipulate data
#   train_tbl <- training(split) %>%
#     add_column(key = "training")
# 
#   test_tbl  <- testing(split) %>%
#     add_column(key = "testing")
# 
#   data_manipulated <- bind_rows(train_tbl, test_tbl) %>%
#     as_tbl_time(index = index) %>%
#     mutate(key = fct_relevel(key, "training", "testing"))
# 
#   # Collect attributes
#   train_time_summary <- train_tbl %>%
#     tk_index() %>%
#     tk_get_timeseries_summary()
# 
#   test_time_summary <- test_tbl %>%
#     tk_index() %>%
#     tk_get_timeseries_summary()
# 
#   # Visualize
#   g <- data_manipulated %>%
#     ggplot(aes(x = index, y = value, color = key)) +
#     geom_line(size = size, alpha = alpha) +
#     theme_tq(base_size = base_size) +
#     scale_color_tq() +
#     labs(
#       #title    = glue("Split: {split$id}"),
#       subtitle = glue("{train_time_summary$start} to {test_time_summary$end}"),
#       y = "", x = ""
#     ) +
#     theme(legend.position = "none")
# 
#   if (expand_y_axis) {
# 
#     series_time_summary <- social_benefits_expenditure %>%
#       tk_index() %>%
#       tk_get_timeseries_summary()
# 
#     g <- g +
#       scale_x_date(limits = c(series_time_summary$start,
#                               series_time_summary$end))
#   }
# 
#   return(g)
# }
# 
# # Plottinng one split only
# rolling_origin_resamples$splits[[5]] %>%
#   plot_split(expand_y_axis = TRUE) #+ theme(legend.position = "top")

  

# 
# 
# # Plotting function that scales to all splits 
# plot_sampling_plan <- function(sampling_tbl, expand_y_axis = TRUE, 
#                                ncol = 3, alpha = 1, size = 1, base_size = 14, 
#                                title = "Sampling Plan") {
#   
#   # Map plot_split() to sampling_tbl
#   sampling_tbl_with_plots <- sampling_tbl %>%
#     mutate(gg_plots = map(splits, plot_split, 
#                           expand_y_axis = expand_y_axis,
#                           alpha = alpha, base_size = base_size))
#   
#   # Make plots with cowplot
#   plot_list <- sampling_tbl_with_plots$gg_plots 
#   
#   p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
#   legend <- get_legend(p_temp)
#   
#   p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
#   
#   p_title <- ggdraw() + 
#     draw_label(title, size = 18, fontface = "bold", colour = palette_light()[[1]])
#   
#   g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
#   
#   return(g)
#   
# }
# 
# rolling_origin_resamples %>%
#   plot_sampling_plan(expand_y_axis = TRUE, ncol = 3, alpha = 1, size = 1, base_size = 10, 
#                      title = "Backtesting Strategy: Rolling Origin Sampling Plan")
# 
# rolling_origin_resamples %>%
#   plot_sampling_plan(expand_y_axis = FALSE, ncol = 3, alpha = 1, size = 1, base_size = 10, 
#                      title = "Backtesting Strategy: Zoomed In")

# LSTM Keras neural network implementation 
predict_keras_lstm <- function(split, epochs = 300, n_units = 50, batch_size = 6, tsteps = 24, n_layers = 3, ...) {
  
  lstm_prediction <- function(split, epochs, n_units, batch_size, tsteps, n_layers, ...) {
    
    # Data Setup
    df_trn <- training(split)
    df_tst <- testing(split)
    
    df <- bind_rows(
      df_trn %>% add_column(key = "training"),
      df_tst %>% add_column(key = "testing")) %>% 
      as_tbl_time(index = index)
    
    # Preprocessing
    rec_obj <- recipe(value ~ ., df) %>%
      step_sqrt(value) %>%
      step_center(value) %>%
      step_scale(value) %>%
      prep()
    
    df_processed_tbl <- bake(rec_obj, df)
    
    # Save center and scale
    center_history <- rec_obj$steps[[2]]$means["value"]
    scale_history  <- rec_obj$steps[[3]]$sds["value"]
    
    # LSTM Plan
    lag_setting  <- nrow(df_tst) #=12
    batch_size   <- batch_size
    train_length <- nrow(df_trn) #=120
    tsteps       <- tsteps
    epochs       <- epochs
    
    # Train/Test Setup (3D arary - predictors and 2D array - target)
    lag_train_tbl <- df_processed_tbl %>%
      mutate(value_lag = lag(value, n = lag_setting)) %>%
      filter(!is.na(value_lag)) %>%
      filter(key == "training") %>%
      tail(train_length)
    
    x_train_vec <- lag_train_tbl$value_lag
    x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), tsteps, 1))
    
    y_train_vec <- lag_train_tbl$value
    y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), tsteps))
    
    lag_test_tbl <- df_processed_tbl %>%
      mutate(
        value_lag = lag(value, n = lag_setting)
      ) %>%
      filter(!is.na(value_lag)) %>%
      filter(key == "testing")
    
    x_test_vec <- lag_test_tbl$value_lag
    x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), tsteps, 1))
    
    y_test_vec <- lag_test_tbl$value
    y_test_arr <- array(data = y_test_vec, dim = c(length(y_test_vec), tsteps))
    
    # LSTM Model
    model <- keras_model_sequential()
    
    if (n_layers == 2) {
      model %>%
        layer_lstm(units            = n_units, 
                   input_shape      = c(tsteps, 1), 
                   batch_size       = batch_size,
                   return_sequences = TRUE, 
                   stateful         = TRUE) %>%
        #dropout          = 0.1,
        #recurrent_dropout= 0.1) %>% 
        # layer_lstm(units            = n_units,
        #            batch_size       = batch_size,
        #            return_sequences = TRUE,
        #            stateful         = TRUE) %>%
        # layer_lstm(units            = n_units,
        #            batch_size       = batch_size,
        #            return_sequences = TRUE,
        #            stateful         = TRUE) %>%
        layer_lstm(units            = n_units, 
                   return_sequences = FALSE, 
                   stateful         = TRUE) %>% 
        #dropout          = 0.1,
        #recurrent_dropout= 0.1
        layer_dense(units = tsteps)
      #layer_dense(units = 1)
    } 
    else { #n_layers=3
      model %>%
        layer_lstm(units            = n_units, 
                   input_shape      = c(tsteps, 1), 
                   batch_size       = batch_size,
                   return_sequences = TRUE, 
                   stateful         = TRUE) %>%
        #dropout          = 0.1,
        #recurrent_dropout= 0.1) %>% 
        layer_lstm(units            = n_units,
                   batch_size       = batch_size,
                   return_sequences = TRUE,
                   stateful         = TRUE) %>%
        # layer_lstm(units            = n_units,
        #            batch_size       = batch_size,
        #            return_sequences = TRUE,
        #            stateful         = TRUE) %>%
        layer_lstm(units            = n_units, 
                   return_sequences = FALSE, 
                   stateful         = TRUE) %>% 
        #dropout          = 0.1,
        #recurrent_dropout= 0.1
        layer_dense(units = tsteps)
        #layer_dense(units = 1)
        #time_distributed(layer_dense(units = 1))
    }
      
    # Compile model
    model %>% 
      compile(loss = 'mae', optimizer = 'adam')
    
    # Fitting LSTM
    for (i in 1:epochs) {
      model %>% fit(x = x_train_arr, 
                    y = y_train_arr, 
                    batch_size = batch_size,
                    epochs     = 1, 
                    verbose    = 1, 
                    shuffle    = FALSE)
      
      model %>% reset_states()
      cat("Epoch: ", i)
    }
    
    #plot(history)
    
    # Predict and Return Tidy Data
    pred_out <- model %>% 
      predict(x_test_arr, batch_size = batch_size) %>%
      .[,1] 
    
    # Retransform values
    pred_tbl <- tibble(
      index   = lag_test_tbl$index,
      value   = (pred_out * scale_history + center_history)^2
    ) 
    
    # Combine actual data with predictions
    tbl_1 <- df_trn %>%
      add_column(key = "actual")
    
    tbl_2 <- df_tst %>%
      add_column(key = "actual")
    
    tbl_3 <- pred_tbl %>%
      add_column(key = "predict")
    
    # Create time_bind_rows() to solve dplyr issue
    time_bind_rows <- function(data_1, data_2, index) {
      index_expr <- enquo(index)
      bind_rows(data_1, data_2) %>%
        as_tbl_time(index = !! index_expr)
    }
    
    ret <- list(tbl_1, tbl_2, tbl_3) %>%
      reduce(time_bind_rows, index = index) %>%
      arrange(key, index) %>%
      mutate(key = as_factor(key))
    
    return(ret)
    
  }
  
  safe_lstm <- possibly(lstm_prediction, otherwise = NA)
  
  #safe_lstm(split, epochs, n_units, ...)
  safe_lstm(split, epochs, n_units, batch_size, tsteps, n_layers, ...)
  
  
}

# Call LSTM Keras implementation passing the parameters (n_layer = {2,3})
sample_predictions_lstm_tbl <- rolling_origin_resamples %>%
  mutate(predict = map(splits, predict_keras_lstm, epochs = 100, n_units = 50, batch_size = 6, tsteps = 24, n_layers = 3))

 sample_predictions_lstm_tbl


#Assessing the prediction error (RMSE)
calc_rmse <- function(prediction_tbl) {
  
  rmse_calculation <- function(data) {
    data %>%
      spread(key = key, value = value) %>%
      select(-index) %>%
      filter(!is.na(predict)) %>%
      rename(truth=actual,estimate=predict) %>%
      rmse(truth, estimate)
  }
  
  safe_rmse <- possibly(rmse_calculation, otherwise = NA)
  
  safe_rmse(prediction_tbl)
}

sample_rmse_tbl <- sample_predictions_lstm_tbl %>%
  mutate(rmse = map_dbl(predict, calc_rmse)) %>%
  select(id, rmse)

sample_rmse_tbl

sample_rmse_tbl %>%
  summarize(
    mean_rmse = mean(rmse),
    sd_rmse   = sd(rmse)
  )



# Setup single plot function
plot_prediction <- function(data, id, alpha = 1, size = 2, base_size = 14) {
  
  rmse_val <- calc_rmse(data)
  
  g <- data %>%
    ggplot(aes(index, value, color = key)) +
    geom_point(alpha = alpha, size = size) + 
    theme_tq(base_size = base_size) +
    scale_color_tq() +
    theme(legend.position = "none") +
    labs(
      title = glue("{id}, RMSE: {round(rmse_val, digits = 1)}"),
      x = "", y = ""
    )
  
  return(g)
}


# plot all splits
plot_predictions <- function(sampling_tbl, predictions_col, 
                             ncol = 3, alpha = 1, size = 2, base_size = 14,
                             title = "Backtested Predictions") {
  
  predictions_col_expr <- enquo(predictions_col)
  
  # Map plot_split() to sampling_tbl
  sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(gg_plots = map2(!! predictions_col_expr, id, 
                           .f        = plot_prediction, 
                           alpha     = alpha, 
                           size      = size, 
                           base_size = base_size)) 
  
  # Make plots with cowplot
  plot_list <- sampling_tbl_with_plots$gg_plots 
  
  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)
  
  p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
  
  
  
  p_title <- ggdraw() + 
    draw_label(title, size = 18, fontface = "bold", colour = palette_light()[[1]])
  
  g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
  
  return(g)
  
}


sample_predictions_lstm_tbl %>%
  plot_predictions(predictions_col = predict, alpha = 0.5, size = 1, base_size = 10,
                   title = "Keras Stateful LSTM: Backtested Predictions")

