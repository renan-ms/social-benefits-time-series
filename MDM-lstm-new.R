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
library(tfruns)

library(rlang)

# sun_spots <- datasets::sunspot.month %>%
#   tk_tbl() %>%
#   mutate(index = as_date(index)) %>%
#   as_tbl_time(index = index)
# 
# sun_spots
# 
# # Set backtesting strategy (cross validation - splits)
# periods_train <- 12 * 100
# periods_test  <- 12 * 50
# skip_span     <- 12 * 22 - 1
# 
# set.seed(772018)
# 
# rolling_origin_resamples <- rolling_origin(
#   sun_spots,
#   initial    = periods_train,
#   assess     = periods_test,
#   cumulative = FALSE,
#   skip       = skip_span
# )
# 
# rolling_origin_resamples

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

set.seed(772018)

rolling_origin_resamples <- rolling_origin(
  social_benefits_expenditure,
  initial    = periods_train,
  assess     = periods_test,
  cumulative = FALSE,
  skip       = skip_span
)

rolling_origin_resamples

# functions used



obtain_predictions <- function(split) {
  df_trn <- analysis(split)[1:90, , drop = FALSE]
  #df_trn <- analysis(split)[1:180, , drop = FALSE]
  df_val <- analysis(split)[91:120, , drop = FALSE]
  df_tst <- assessment(split)

  
  df <- bind_rows(
    df_trn %>% add_column(key = "training"),
    df_val %>% add_column(key = "validation"),
    df_tst %>% add_column(key = "testing")) %>%
    as_tbl_time(index = index)
  
  rec_obj <- recipe(value ~ ., df) %>%
    step_sqrt(value) %>%
    step_center(value) %>%
    step_scale(value) %>%
    prep()
  
  df_processed_tbl <- bake(rec_obj, df)
  
  center_history <- rec_obj$steps[[2]]$means["value"]
  scale_history  <- rec_obj$steps[[3]]$sds["value"]
  
  FLAGS <- flags(
    flag_boolean("stateful", TRUE),
    flag_boolean("stack_layers", TRUE),
    flag_integer("batch_size", 6),
    flag_integer("n_timesteps", 3),
    flag_integer("n_epochs", 600),
    flag_numeric("dropout", 0.2),
    flag_numeric("recurrent_dropout", 0.2),
    flag_string("loss", "logcosh"),
    flag_string("optimizer_type", "sgd"),
    #flag_string("loss", "mae"),
    #flag_string("optimizer_type", "adam"),
    flag_integer("n_units", 64),
    flag_numeric("lr", 0.001),
    flag_numeric("momentum", 0.9),
    flag_integer("patience", 10)
  )
  
  n_predictions <- FLAGS$n_timesteps
  n_features <- 1
  
  optimizer <- switch(FLAGS$optimizer_type,
                      sgd = optimizer_sgd(lr = FLAGS$lr, momentum = FLAGS$momentum),
                      adam = optimizer_adam(lr = FLAGS$lr))
  callbacks <- list(
    callback_early_stopping(patience = FLAGS$patience)
  )
  
  train_vals <- df_processed_tbl %>%
    filter(key == "training") %>%
    select(value) %>%
    pull()
  valid_vals <- df_processed_tbl %>%
    filter(key == "validation") %>%
    select(value) %>%
    pull()
  test_vals <- df_processed_tbl %>%
    filter(key == "testing") %>%
    select(value) %>%
    pull()
  
  
  build_matrix <- function(tseries, overall_timesteps) {
    t(sapply(1:(length(tseries) - overall_timesteps + 1), function(x) 
      tseries[x:(x + overall_timesteps - 1)]))
  }
  
  reshape_X_3d <- function(X) {
    dim(X) <- c(dim(X)[1], dim(X)[2], 1)
    X
  }
  
  train_matrix <- build_matrix(train_vals, FLAGS$n_timesteps + n_predictions)
  valid_matrix <- build_matrix(valid_vals, FLAGS$n_timesteps + n_predictions)
  test_matrix <- build_matrix(test_vals, FLAGS$n_timesteps + n_predictions)
  
  X_train <- train_matrix[, 1:FLAGS$n_timesteps]
  y_train <- train_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
  X_train <- X_train[1:(nrow(X_train) %/% FLAGS$batch_size * FLAGS$batch_size),]
  y_train <- y_train[1:(nrow(y_train) %/% FLAGS$batch_size * FLAGS$batch_size),]

  X_valid <- valid_matrix[, 1:FLAGS$n_timesteps]
  y_valid <- valid_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
  X_valid <- X_valid[1:(nrow(X_valid) %/% FLAGS$batch_size * FLAGS$batch_size),]
  y_valid <- y_valid[1:(nrow(y_valid) %/% FLAGS$batch_size * FLAGS$batch_size),]
  
  X_test <- test_matrix[, 1:FLAGS$n_timesteps]
  y_test <- test_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
  X_test <- X_test[1:(nrow(X_test) %/% FLAGS$batch_size * FLAGS$batch_size),]
  y_test <- y_test[1:(nrow(y_test) %/% FLAGS$batch_size * FLAGS$batch_size),]

  
  X_train <- reshape_X_3d(X_train)
  X_valid <- reshape_X_3d(X_valid)
  X_test <- reshape_X_3d(X_test)
  
  y_train <- reshape_X_3d(y_train)
  y_valid <- reshape_X_3d(y_valid)
  y_test <- reshape_X_3d(y_test)
  
  model <- keras_model_sequential()
  
  model %>%
    layer_lstm(
      units = FLAGS$n_units,
      batch_input_shape  = c(FLAGS$batch_size, FLAGS$n_timesteps, n_features),
      dropout = FLAGS$dropout,
      recurrent_dropout = FLAGS$recurrent_dropout,
      return_sequences = TRUE,
      stateful = FLAGS$stateful
    )
  
  if (FLAGS$stack_layers) {
    model %>%
      layer_lstm(
        units   = FLAGS$n_units,
        dropout = FLAGS$dropout,
        recurrent_dropout = FLAGS$recurrent_dropout,
        return_sequences = TRUE,
        stateful = FLAGS$stateful
      ) %>%
      layer_lstm(
        units   = FLAGS$n_units,
        dropout = FLAGS$dropout,
        recurrent_dropout = FLAGS$recurrent_dropout,
        return_sequences = TRUE,
        stateful = FLAGS$stateful)      
  }
  model %>% time_distributed(layer_dense(units = 1))
  
  model %>%
    compile(
      loss = FLAGS$loss,
      optimizer = optimizer,
      #metrics = list("mean_absolute_error")
      metrics = list("mean_squared_error")      
    )
  
  if (!FLAGS$stateful) {
    model %>% fit(
      x          = X_train,
      y          = y_train,
      validation_data = list(X_valid, y_valid),
      batch_size = FLAGS$batch_size,
      epochs     = FLAGS$n_epochs,
      callbacks = callbacks
    )
    
  } else {
    for (i in 1:FLAGS$n_epochs) {
      model %>% fit(
        x          = X_train,
        y          = y_train,
        validation_data = list(X_valid, y_valid),
        callbacks = callbacks,
        batch_size = FLAGS$batch_size,
        epochs     = 1,
        shuffle    = FALSE
      )
      model %>% reset_states()
    }
  }
  
  if (FLAGS$stateful)
    model %>% reset_states()  
  
  # model <- keras_model_sequential()
  # 
  # model %>%
  #   layer_lstm(
  #     units              = FLAGS$n_units,
  #     batch_input_shape  = c(FLAGS$batch_size, FLAGS$n_timesteps, n_features),
  #     dropout            = FLAGS$dropout,
  #     recurrent_dropout  = FLAGS$recurrent_dropout,
  #     return_sequences   = TRUE
  #   ) %>% time_distributed(layer_dense(units = 1))
  # 
  # model %>%
  #   compile(
  #     loss      = FLAGS$loss,
  #     optimizer = optimizer,
  #     metrics   = list("mean_squared_error")
  #   )
  # 
  # model %>% fit(
  #   x               = X_train,
  #   y               = y_train,
  #   #validation_data = list(X_valid, y_valid),
  #   batch_size      = FLAGS$batch_size,
  #   epochs          = FLAGS$n_epochs
  #   #callbacks       = callbacks
  # )
  
  
  pred_train <- model %>%
    predict(X_train, batch_size = FLAGS$batch_size) %>%
    .[, , 1]
  
  # Retransform values
  pred_train <- (pred_train * scale_history + center_history) ^ 2
  compare_train <- df %>% filter(key == "training")
  
  for (i in 1:nrow(pred_train)) {
    varname <- paste0("pred_train", i)
    compare_train <-
      mutate(compare_train, !!varname := c(
        rep(NA, FLAGS$n_timesteps + i - 1),
        pred_train[i, ],
        rep(NA, nrow(compare_train) - FLAGS$n_timesteps * 2 - i + 1)
      ))
  }
  
  pred_test <- model %>%
    predict(X_test, batch_size = FLAGS$batch_size) %>%
    .[, , 1]
  
  # Retransform values
  pred_test <- (pred_test * scale_history + center_history) ^ 2
  compare_test <- df %>% filter(key == "testing")
  
  for (i in 1:nrow(pred_test)) {
    varname <- paste0("pred_test", i)
    compare_test <-
      mutate(compare_test, !!varname := c(
        rep(NA, FLAGS$n_timesteps + i - 1),
        pred_test[i, ],
        rep(NA, nrow(compare_test) - FLAGS$n_timesteps * 2 - i + 1)
      ))
  }
  list(train = compare_train, test = compare_test)
  
}

# CHAMA AQUI
all_split_preds <- rolling_origin_resamples %>%
  mutate(predict = map(splits, obtain_predictions))




#RMSE on all splits
calc_rmse <- function(df) {
  coln <- colnames(df)[4:ncol(df)]
  cols <- map(coln, quo(sym(.)))
  map_dbl(cols, function(col)
    rmse(
      df,
      truth = value,
      estimate = !!col,
      na.rm = TRUE
    )) %>% mean()
}


all_split_preds <- all_split_preds %>% unnest(predict)
all_split_preds_train <- all_split_preds[seq(1, 21, by = 2), ]
all_split_preds_test <- all_split_preds[seq(2, 22, by = 2), ]

#all_split_preds_train <- all_split_preds[seq(1, 21, by = 2), ]
#all_split_preds_test <- all_split_preds[seq(2, 22, by = 2), ]

all_split_rmses_train <- all_split_preds_train %>%
  mutate(rmse = map_dbl(predict, calc_rmse)) %>%
  select(id, rmse)

all_split_rmses_test <- all_split_preds_test %>%
  mutate(rmse = map_dbl(predict, calc_rmse)) %>%
  select(id, rmse)

