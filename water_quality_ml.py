# Importing Dependencies
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from sparkxgb import XGBoostRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark session
spark = SparkSession.builder.appName('Water Quality Prediction').getOrCreate()

# Load the dataset
#hdfs_path = "hdfs://192.168.100.86/user/cloudera/projet/water_quality/water_quality_data.csv"
#df = spark.read.csv(hdfs_path, header=True, inferSchema=True)
df = spark.read.csv(r"C:\Users\user\Desktop\PFE\water_data\water_quality_data.csv", header=True, inferSchema=True)

# Show the first 5 rows of dataset
df.show(5)
df.printSchema()

# Statistics
summary = df.describe().toPandas().transpose()
print(summary)

# Correlation Matrix
selected_columns = ['Conductivity', 'PH_of_the_solution', 'PH_of_water_backwash', 'PH_of_seawater', 
                'Pressure_of_water_entering_membrane_1', 'Pressure_of_water_entering_membrane_2', 
                'Pressure_of_water_entering_membrane_3', 'Pressure_of_water_entering_membrane_4', 
                'Pressure_of_water_entering_membrane_5', 'Turbidity', 'Temperature', 'ph_of_treated_water']

correlation_matrix = df.select(selected_columns).toPandas().corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Check for missing values in all columns
missing_values_all = df.filter(
    col('Conductivity').isNull() |
    col('PH_of_the_solution').isNull() |
    col('PH_of_water_backwash').isNull() |
    col('PH_of_seawater').isNull() |
    col('Pressure_of_water_entering_membrane_1').isNull() |
    col('Pressure_of_water_entering_membrane_2').isNull() |
    col('Pressure_of_water_entering_membrane_3').isNull() |
    col('Pressure_of_water_entering_membrane_4').isNull() |
    col('Pressure_of_water_entering_membrane_5').isNull() |
    col('Turbidity').isNull() |
    col('Temperature').isNull() |
    col('ph_of_treated_water').isNull()
)
# Count the number of missing values
num_missing_values_all = missing_values_all.count()
print(f" Number of rows with missing values is : {num_missing_values_all}")

# Remove rows with missing values
df = df.na.drop(subset=['Conductivity', 'PH_of_the_solution', 'PH_of_water_backwash', 'PH_of_seawater','Pressure_of_water_entering_membrane_1', 'Pressure_of_water_entering_membrane_2', 'Pressure_of_water_entering_membrane_3', 'Pressure_of_water_entering_membrane_4','Pressure_of_water_entering_membrane_5', 'Turbidity', 'Temperature', 'ph_of_treated_water'])
df.show()

# Define X and y:
feature_columns = ['Conductivity', 'PH_of_the_solution', 'PH_of_water_backwash', 'PH_of_seawater','Pressure_of_water_entering_membrane_1', 'Pressure_of_water_entering_membrane_2', 'Pressure_of_water_entering_membrane_3', 'Pressure_of_water_entering_membrane_4','Pressure_of_water_entering_membrane_5', 'Turbidity', 'Temperature']
assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features")
data = assembler.transform(df)

# split the data using a ratio of 90% train set:
train_ratio = 0.9
test_ratio = 1 - train_ratio
seed = 123
train_data, test_data = data.randomSplit([train_ratio, test_ratio], seed=seed)
####################################################### MODELS ############################################################################
# Create and train Random Forest model
rf = RandomForestRegressor(featuresCol="features", labelCol="PH_of_treated_water", numTrees=100)
rf_pipeline = Pipeline(stages=[rf])
rf_model = rf_pipeline.fit(train_data)
rf_predictions = rf_model.transform(test_data)

# Evaluate model
evaluator = RegressionEvaluator(labelCol="PH_of_treated_water", predictionCol="prediction", metricName="rmse")
rf_rmse = evaluator.evaluate(rf_predictions)
rf_mse = evaluator.evaluate(rf_predictions, {evaluator.metricName: "mse"})
rf_r2 = evaluator.evaluate(rf_predictions, {evaluator.metricName: "r2"})
#################################################
# Create and train Gradient-Boosted Trees model
gbt = GBTRegressor(featuresCol="scaled_features", labelCol="PH_of_treated_water")
gbt_pipeline = Pipeline(stages=[gbt])
gbt_model = gbt_pipeline.fit(train_data)
gbt_predictions = gbt_model.transform(test_data)

# Evaluate model
gbt_rmse = evaluator.evaluate(gbt_predictions)
gbt_mse = evaluator.evaluate(gbt_predictions, {evaluator.metricName: "mse"})
gbt_r2 = evaluator.evaluate(gbt_predictions, {evaluator.metricName: "r2"})
##################################################
# Create and train Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="PH_of_treated_water")
lr_pipeline = Pipeline(stages=[lr])
lr_model = lr.fit(train_data)
lr_predictions = lr_model.transform(test_data)

# Evaluate model
lr_rmse = evaluator.evaluate(lr_predictions)
lr_mse = evaluator.evaluate(lr_predictions, {evaluator.metricName: "mse"})
lr_r2 = evaluator.evaluate(lr_predictions, {evaluator.metricName: "r2"})
####################################################### EVALUATE MODELS ############################################################################

print(f"Random Forest RMSE: {rf_rmse}")
print(f"Mean Squared Error (MSE): {rf_mse}")
print(f"R-squared on test data: {rf_r2}")

print(f"Random Forest RMSE: {gbt_rmse}")
print(f"Mean Squared Error (MSE):{gbt_mse}")
print(f"R-squared on test data:{gbt_r2}")

print(f"Random Forest RMSE: {lr_rmse}")
print(f"Mean Squared Error (MSE):{lr_mse}")
print(f"R-squared on test data:{lr_r2}")

####################################################### CHOOSE BEST MODEL #########################################################################
model_metrics = {
    "Random Forest": rf_r2,
    "Gradient-Boosted Trees": gbt_r2,
    "Linear Regression": lr_r2
}

best_model_name = max(model_metrics, key=model_metrics.get)
best_model = {
    "Random Forest": rf_model,
    "Gradient-Boosted Trees": gbt_model,
    "Linear Regression": lr_model
}[best_model_name]

print(f"Best Model: {best_model_name} with R2: {model_metrics[best_model_name]}")

####################################################### SAVE MODEL ############################################################################

# Save the model:
model_path = r"C:\Users\user\Desktop\PFE\WaterQuality"
best_model.write().overwrite().save(model_path)

# Stop the Spark session
spark.stop()