##########################################################################
# PROYECTO: Entrega Final Ciencia de Datos
# AUTOR: Raúl Fauste
##########################################################################

##########################################################################
# CONFIGURACIONES INICIALES

setwd("C:/Users/raulfauste/Desktop/R")
set.seed(11)

library(caret)
library(fastDummies)
library(discretization)
library(nnet)
library(e1071)
library(kernlab)
library(MLmetrics)
##########################################################################

##########################################################################
# LECTURA DE INSTANCIAS

train <- read.table("train.csv", header = TRUE, sep = ",", quote = "")
test  <- read.table("test.csv", header = TRUE, sep = ",", quote = "")

# Dividimos en entrenamiento y validacion
train$Exited <- as.factor(train$Exited) 

division <- createDataPartition(train$Exited, p = 0.8, list = FALSE)

validation  <- train[-division, ]
train <- train[division, ]
##########################################################################

##########################################################################
# FUNCIONES AUXILIARES

# Limpieza
limpieza <- function(df, missingV) {
  df <- df[, !(names(df) %in% c("RowNumber", "CustomerId", "Surname"))]
  
  df$HasCrCard[is.na(df$HasCrCard)] <- 1
  
  df$CreditScore[is.na(df$CreditScore)] <- missingV[[1]]
  df$Balance[is.na(df$Balance)] <- missingV[[2]]
  df$EstimatedSalary[is.na(df$EstimatedSalary)] <- missingV[[3]]
  
  indices_na <- which(is.na(df$NumOfProducts))
  N <- length(indices_na)
  if(N > 0) {
    df$NumOfProducts[indices_na] <- sample(as.numeric(names(missingV[[4]])), 
                                           size = N, 
                                           replace = TRUE, 
                                           prob = missingV[[4]])
  }
  
  df$HasCrCard <- as.numeric(df$HasCrCard)
  df$IsActiveMember <- as.numeric(df$IsActiveMember)
  
  df$Gender <- ifelse(df$Gender == "Male", 1, 0)
  
  df$Geography <- factor(df$Geography, levels = c("France", "Germany", "Spain"), labels = c("F", "G", "S"))
  df <- dummy_cols(df, select_columns = "Geography", remove_first_dummy = TRUE, remove_selected_columns = TRUE)
  
  return(df)
}


# Discretizacion
discretizar <- function(df, res_mdlp) {
  df$Age <- cut(df$Age, breaks = c(-Inf, res_mdlp$cutp[[1]], Inf), labels = FALSE)
  df$CreditScore <- cut(df$CreditScore, breaks = c(-Inf, res_mdlp$cutp[[2]], Inf), labels = FALSE)
  df$Balance <- cut(df$Balance, breaks = c(-Inf, res_mdlp$cutp[[3]], Inf), labels = FALSE)
  df$NumOfProducts <- cut(df$NumOfProducts, breaks = c(-Inf, res_mdlp$cutp[[4]], Inf), labels = FALSE)
  
  return(df)
}


# Estandarizacion
estandarizar <- function(df, parameters) {
  df <- predict(parameters, df)
  
  return(df)
}


# Entrenamiento
entrenar <- function(datos, metodo) {
  control <- trainControl(method = "cv", number = 10,  classProbs = TRUE, summaryFunction = prSummary)
  
  if(metodo == "nnet") {
    grid <- expand.grid(size = c(3, 5), decay = c(0.1, 0.5, 1))
    modelo <- train(Exited ~ ., data = datos, method = metodo, 
                    trace = FALSE, tuneGrid = grid, trControl = control)
  } else {
    modelo <- train(Exited ~ ., data = datos, method = metodo, trControl = control)
  }
  
  return(modelo)
}


# Evaluacion
evaluar_modelo <- function(datos, modelo) {
  
  datos$Exited <- factor(ifelse(datos$Exited == 1, "True", "False"),
                              levels = c("True", "False"))
  
  preds <- predict(modelo, datos)
  print(confusionMatrix(preds, datos$Exited, mode = "everything", positive = "True"))
}


# Exportacion
exportar_kaggle <- function(datos, modelo, nombre_archivo) {
  probs <- predict(modelo, datos, type = "prob")
  preds_final <- ifelse(probs$True > 0.45, 1, 0)
  
  test_original <- read.table("test.csv", header = TRUE, sep = ",", quote = "")
  
  entrega <- data.frame(
    CustomerId = test_original$CustomerId, 
    Exited = preds_final
  )
  
  write.csv(entrega, nombre_archivo, row.names = FALSE, quote = FALSE)
}

# Procesado (Completo)
procesado <- function(train, test, validation) {
  media_credit <- mean(train$CreditScore, na.rm = TRUE)
  media_balance <- mean(train$Balance, na.rm = TRUE)
  media_salary <- mean(train$EstimatedSalary, na.rm = TRUE)
  dist_products <- prop.table(table(train$NumOfProducts))
  
  missing <- list(media_credit, media_balance, media_salary, dist_products)
  
  train$Exited <- factor(ifelse(train$Exited == 1, "True", "False"),
                         levels = c("True", "False"))
  
  train <- limpieza(train, missing)
  validation <- limpieza(validation, missing)
  test  <- limpieza(test, missing)
  
  res_mdlp <- mdlp(train[, c("Age", "CreditScore", "Balance", "NumOfProducts", "Exited")])
  
  train <- discretizar(train, res_mdlp)
  validation <- discretizar(validation, res_mdlp)
  test  <- discretizar(test, res_mdlp)
  
  params <- preProcess(train[, c("Tenure", "EstimatedSalary")], method = c("center", "scale"))
  
  train <- estandarizar(train, params)
  validation <- estandarizar(validation, params)
  test <- estandarizar(test, params)
  
  return(list(train,test,validation))
}

##########################################################################

##########################################################################
# MAIN

instancias <- procesado(train,test,validation)

train <- instancias[[1]]
test <- instancias[[2]]
validation <- instancias[[3]]

model_nn  <- entrenar(train, "nnet")
model_nb  <- entrenar(train, "nb")
model_dt <- entrenar(train, "rpart")
model_knn <- entrenar(train, "knn")

evaluar_modelo(validation, model_nn)
evaluar_modelo(validation, model_nb)
evaluar_modelo(validation, model_dt)
evaluar_modelo(validation, model_knn)

exportar_kaggle(test, model_nn, "entrega_final_raul_fauste.csv")

##########################################################################

