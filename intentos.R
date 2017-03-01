# En este script pongo todo lo que voy haciendo antes de pasarlo a 
# R markdown. Lo que vaya funcionando y ss explicaciones en limpio (y en inglés)
# las pasaré al archivo proyecto.Rmd

# Solo consultar y modificar si quires revisar el orden en que se hicieron las cosas


# Descargar datos de la página y cargar data frames

if (!dir.exists("pml-data")){
    dir.create("pml-data")
    if (!file.exists("pml-data/pml-training.csv"))
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                      "pml-data/pml-training.csv")
    if (!file.exists("pml-data/pml-testing.csv"))
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                      "pml-data/pml-testing.csv")
}

pml.train <- read.csv("pml-data/pml-training.csv", 
                      na.strings = c('', 'NA', '"NA"', "#DIV/0"))
pml.test <- read.csv("pml-data/pml-testing.csv", 
                     na.strings = c('', 'NA', '"NA"', "#DIV/0"))

library(caret)

nzv <- nearZeroVar(pml.train)
pml.train <- pml.train[, -nzv]
pml.test <- pml.test[, -nzv]

# Elimina las columnas con más de 50% de valores faltantes
a <- sapply(pml.train, function (l) sum(is.na(l))/19622 < 0.5)
pml.train <- pml.train[, a]
pml.test <- pml.test[, a]

# Revisa si quedaron valores faltantes
which(is.na(pml.train))

# Usa el nombre como factor, así como la clase a predecir
pml.train$user_name <- factor(pml.train$user_name)
pml.train$classe <- factor(pml.train$classe)
pml.test$user_name <- factor(pml.test$user_name)

# Sin hacer trampa con la serie de tiempo
pml.train0 <- pml.train[, -c(1,3,4,5)]
pml.test0 <- pml.test[, -c(1,3,4,5)]

des.cor <- cor(pml.train0[,2:54])
highly.cor <- findCorrelation(des.cor, cutoff = .9) + 1
pml.train0 <- pml.train0[, -highly.cor]
pml.test0 <- pml.test0[, -highly.cor]

# Centrado y escalado
cen.esc <- preProcess(pml.train0, method = c("center", "scale"))
pml.train1 <- predict(cen.esc, pml.train0)
pml.test1 <- predict(cen.esc, pml.test0)

# Componentes principales
# trans <- preProcess(pml.train1, method='pca')
# pml.train2 <- predict(trans, pml.train1)
# pml.test2 <- predict(trans, pml.test1)
# 
# 
# featurePlot(x = pml.train1[, 3:6], 
#             y = pml.train1$classe,
#             plot = "pairs", 
#             ## Pass in options to xyplot() to 
#             ## make it prettier
#             scales = list(x = list(relation="free"), 
#                           y = list(relation="free")))
# 
# featurePlot(x = pml.train1[, 3:6], 
#             y = pml.train1$user_name,
#             plot = "pairs", 
#             ## Pass in options to xyplot() to 
#             ## make it prettier
#             scales = list(x = list(relation="free"), 
#                           y = list(relation="free")))

# Split training data between training and validation
set.seed(12345)
inTraining <- createDataPartition(pml.train1$classe, 
                                  p = .75, list = FALSE)
training <- pml.train1[inTraining,]
validation <- pml.train1[-inTraining,]

# Using 3-fold-cross-validation 
fitControl <- trainControl(method = "cv",
                           number = 10)

# Using a random forest model
set.seed(54321)
mdl1 <- train(classe ~ ., data = training, 
              method = "rf", 
              trControl = fitControl,
              verbose = FALSE)

set.seed(54321)
mdl2 <- train(classe ~ ., data = training, 
              method = "gbm", 
              trControl = fitControl,
              verbose = FALSE)


# Compare the two models

resamps <- resamples(list(Random_Forest = mdl1,
                          Gradien_Boosting = mdl2))
bwplot(resamps, layout = c(2, 1))


accuracy <- function (model, data)    
    mean(data$classe == predict(model, data)) 
a <- data.frame(in.sample = c(accuracy(mdl1, training),
                              accuracy(mdl2, training)),
                out.sample = c(accuracy(mdl1, validation),
                               accuracy(mdl2, validation)),
                row.names = c('Random Forest', 'Gradient Boosting'))


results <- data.frame(RF= predict(mdl1, pml.test1),
                      GBM = predict(mdl2, pml.test1))

