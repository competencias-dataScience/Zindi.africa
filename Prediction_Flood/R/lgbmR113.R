# load data
load("../newData/dataTrainFeatureEng2.Rdata")
load("../newData/dataTestFeatureEng2.Rdata")

# Selection of variables for analysis 
library(tidyverse)
dataTrain <- dataTrain %>% 
  dplyr::select(long:LC_Type1, rugos_near8, RS1:RS17, wcs)

# Coordinates
library(raster)
library(biomod2)
datosCoord <- data.frame(lon = dataTrain$long,
                         lat = dataTrain$lat,
                         elevation = dataTrain$elevation)
my_raster <- rasterFromXYZ(datosCoord)
crs(my_raster) <- c("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0") 
coordMalawi <- extent(my_raster)

# Raster NDVI
rasterTopography <- raster("../../Datos4_Clima/LandNASA/Topography2000.tif")
rasterTopography <- crop(rasterTopography, coordMalawi)
rta <- dataTrain %>% pull(target_2015)
coord_rta = dataTrain[c('long','lat')]
nombre_rta = "target_2015"

dataModelTopography <- BIOMOD_FormatingData(
  resp.var = rta,
  expl.var = rasterTopography,
  resp.xy = coord_rta,
  resp.name = nombre_rta
)

dataTopography <- dataModelTopography@coord %>% 
  bind_cols(dataModelTopography@data.env.var)

dataTrain <- left_join(dataTrain, dataTopography) %>% 
  rename(Topography = Topography2000)
dataTest$Topography <- dataTrain$Topography

# Evaporation
library(ncdf4)
rasterEvaporation <- raster("../../Datos4_Clima/Copernicus/evaporation/adaptor.mars.internal-1589039948.2857835-13235-23-534e6186-a069-4eb4-a7bd-c82fe398f176.grib")
rta <- dataTrain %>% pull(target_2015)
coord_rta = dataTrain[c('long','lat')]
nombre_rta = "target_2015"

dataModelEvaporation <- BIOMOD_FormatingData(
  resp.var = rta,
  expl.var = rasterEvaporation,
  resp.xy = coord_rta,
  resp.name = nombre_rta
)

dataEvaporation <- dataModelEvaporation@coord %>% 
  bind_cols(dataModelEvaporation@data.env.var)

dataTrain <- left_join(dataTrain, dataEvaporation) %>% 
  rename(evaporation = adaptor.mars.internal.1589039948.2857835.13235.23.534e6186.a069.4eb4.a7bd.c82fe398f176)
dataTest$evaporation <- dataTrain$evaporation

# Humidity 2015-2018
rasterHumid <- raster("../../Datos4_Clima/NASA/humiditySoil/GLDAS_MOS10_M.A201812.001.grb.SUB.grb")
rta <- dataTrain %>% pull(target_2015)
coord_rta = dataTrain[c('long','lat')]
nombre_rta = "target_2015"

dataModelHumid <- BIOMOD_FormatingData(
  resp.var = rta,
  expl.var = rasterHumid,
  resp.xy = coord_rta,
  resp.name = nombre_rta
)

dataHumid <- dataModelHumid@coord %>% 
  bind_cols(dataModelHumid@data.env.var)

dataTrain <- left_join(dataTrain, dataHumid) %>% 
  rename(humid = GLDAS_MOS10_M.A201812.001.grb.SUB)
dataTest$humid <- dataTrain$humid

# Flow water
rasterFlowWater <- raster("../../Datos4_Clima/NASA/FlowWater/baseflow_flux.nc4.nc4")
rta <- dataTrain %>% pull(target_2015)
coord_rta = dataTrain[c('long','lat')]
nombre_rta = "target_2015"

dataModelFlowWater <- BIOMOD_FormatingData(
  resp.var = rta,
  expl.var = rasterFlowWater,
  resp.xy = coord_rta,
  resp.name = nombre_rta
)

dataFlowWater <- dataModelFlowWater@coord %>% 
  bind_cols(dataModelFlowWater@data.env.var)

dataTrain <- left_join(dataTrain, dataFlowWater) %>% 
  rename(flowWater = baseflow_flux)
dataTest$flowWater <- dataTrain$flowWater

# Runoff Flow Surface NASA
rasterrunoffSurface <- raster("../../Datos4_Clima/NASA/runoffSurface/escorrentia_superficial_tormenta.nc4.SUB.nc4")
rta <- dataTrain %>% pull(target_2015)
coord_rta = dataTrain[c('long','lat')]
nombre_rta = "target_2015"

dataModelrunoffSurface<- BIOMOD_FormatingData(
  resp.var = rta,
  expl.var = rasterrunoffSurface,
  resp.xy = coord_rta,
  resp.name = nombre_rta
)

datarunoffSurface<- dataModelrunoffSurface@coord %>% 
  bind_cols(dataModelrunoffSurface@data.env.var)

dataTrain <- left_join(dataTrain, datarunoffSurface) %>% 
  rename(runoffSurface = Storm.surface.runoff)
dataTest$runoffSurface <- dataTrain$runoffSurface

# caret for partition data
library(caret)
set.seed(123)
indx <- createDataPartition(y = dataTrain$target_2015, times = 1, p = 0.70, list = FALSE)
dfTrain <- dataTrain[indx, ]
dfTest <- dataTrain[-indx, ]

# Categorical features
catfeatures <- names(dfTrain %>% select_if(is.factor))

# Data for lightgbm (all less colum 3-->target)
library(lightgbm)
dataTrain_lgbm <- lgb.Dataset(data = data.matrix(dfTrain[, -3]), label = dfTrain[, 3],
                              categorical_feature = catfeatures)
dataTest_lgbm <- lgb.Dataset(data = data.matrix(dfTest[, -3]), label = dfTest[, 3],
                             categorical_feature = catfeatures)

# Parameters for lightgbm
myParams <- list(
  boosting = "gbdt",
  objective = "poisson",
  metric = "rmse",
  learning_rate = 0.005,
  feature_fraction = 1,
  bagging_fraction = 1,
  min_data_in_leaf = 100,
  max_depth = -1
)

# Train model
modelo <- lgb.train(params = myParams,
                    data = dataTrain_lgbm,
                    nrounds = 30000,
                    valids = list(test = dataTest_lgbm),
                    early_stopping_rounds = 500)

# best iter: 13690
# best score: 0.09604879

# Predictions 2019
predicciones <- predict(modelo, data.matrix(dataTest %>%
                                              dplyr::select(long:LC_Type1,
                                                            rugos_near8,
                                                            RS1:RS17, wcs,
                                                            Topography,
                                                            evaporation,
                                                            humid, flowWater,
                                                            runoffSurface)),
                        num_iteration = modelo$best_iter)
x11();hist(predicciones)
predicciones[predicciones > 1] <- 1

# Submission
dataTest %>% 
  dplyr::select(Square_ID) %>% 
  mutate(target = predicciones) ->
  lgbmR113

# Export submission for zindi
write.csv(lgbmR113, file = "Submission/lgbmR113.csv", row.names = FALSE)

# Importance variables
impModelo <- lgb.importance(modelo, percentage = TRUE)
x11()
impModelo %>% 
  ggplot(data = ., aes(x = reorder(Feature,Gain), y = Gain)) +
  coord_flip() + 
  geom_col(color = "black") +
  theme_light()

# Export data for XGBoost H2o
dataTrainH2o <- dataTrain
dataTestH2o <- dataTest %>%
  dplyr::select(long:LC_Type1,
                rugos_near8,
                RS1:RS17, wcs,
                Topography,
                evaporation,
                humid, flowWater,
                runoffSurface,
                Square_ID)
save(dataTrainH2o, file = "../newData/dataTrainh2o.Rdata", compress = "xz")
save(dataTestH2o, file = "../newData/dataTesth2o.Rdata", compress = "xz")
