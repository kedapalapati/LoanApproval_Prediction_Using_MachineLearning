# Databricks notebook source
# MAGIC %md
# MAGIC #                                     Loan Approval Prediction using Machine Learning

# COMMAND ----------

loanDF =spark.read.options(mode='FAILFAST', 
                              header = True, 
                              multiLine=True, 
                              inferSchema=True, 
                              escape='"').csv("/FileStore/tables/credit_risk_dataset.csv")

display(loanDF)

# COMMAND ----------

print(f"Number of rows: {loanDF.count()}")
print(f"Number of columns: {len(loanDF.columns)}")

# COMMAND ----------

loanDF.describe().show()

# COMMAND ----------

#Check for Null Values in Each Column:
from pyspark.sql.functions import col, sum

loanDF.select([sum(col(c).isNull().cast("int")).alias(c) for c in loanDF.columns]).show()

# COMMAND ----------

loanDF.select("loan_status").distinct().show()

# COMMAND ----------

from pyspark.sql.types import IntegerType, FloatType, DoubleType, StringType

# Separate numerical and categorical columns
numerical_columns = [field.name for field in loanDF.schema.fields if isinstance(field.dataType, (IntegerType, FloatType, DoubleType))]
categorical_columns = [field.name for field in loanDF.schema.fields if isinstance(field.dataType, StringType)]

# Display the lists of numerical and categorical columns
print("\nNumerical Columns:", numerical_columns)
print("Categorical Columns:", categorical_columns)

# COMMAND ----------

for col in categorical_columns:
    print(f"\nColumn: {col}")
    unique_values = loanDF.select(col).distinct().rdd.map(lambda row: row[0]).collect()
    print(f"Unique Values: {unique_values}")

# COMMAND ----------

from pyspark.sql.types import IntegerType

# Convert column to IntegerType
loanDF = loanDF.withColumn("person_age", loanDF["person_age"].cast(IntegerType()))

# Verify the data type
print(loanDF.schema["person_age"].dataType)  # Outputs: IntegerType


# COMMAND ----------

from pyspark.sql.functions import when, col

# Binary Encoding for person_home_ownership (e.g., "RENT" = 0, others = 1)
loanDF = loanDF.withColumn(
    "person_home_ownership",
    when(col("person_home_ownership") == "RENT", 0).otherwise(1)
)

# Binary Encoding for cb_person_default_on_file ("No" = 0, "Yes" = 1)
loanDF = loanDF.withColumn(
    "cb_person_default_on_file",
    when(col("cb_person_default_on_file") == "No", 0).otherwise(1)
)

# Show first few rows of updated DataFrame
loanDF.show(5)


# COMMAND ----------

from pyspark.sql.functions import create_map, col, lit
from itertools import chain

# Define an ordinal mapping for loan_grade (A = best, G = worst)
grade_order = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7
}

# Convert dictionary to Spark mapping expression
mapping_expr = create_map([lit(x) for x in chain(*grade_order.items())])

# Apply the mapping to create a numeric column for loan_grade
loanDF = loanDF.withColumn("loan_grade", mapping_expr[col("loan_grade")])

# Show updated DataFrame
loanDF.show(5)


# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# Check unique values in person_home_ownership
distinct_values = loanDF.select("person_home_ownership").distinct().count()

# Always encode loan_intent
indexer2 = StringIndexer(inputCol="loan_intent", outputCol="loan_intent_index")
encoder2 = OneHotEncoder(inputCol="loan_intent_index", outputCol="loan_intent_encoded")

# If person_home_ownership has more than 1 unique value, encode it
stages = [indexer2, encoder2]

if distinct_values > 1:
    indexer1 = StringIndexer(inputCol="person_home_ownership", outputCol="person_home_ownership_index")
    encoder1 = OneHotEncoder(inputCol="person_home_ownership_index", outputCol="person_home_ownership_encoded")
    stages = [indexer1, indexer2, encoder1, encoder2]

# Create and apply pipeline
pipeline = Pipeline(stages=stages)
model = pipeline.fit(loanDF)
loanDF = model.transform(loanDF)

# Drop original columns that were encoded
columns_to_drop = ["loan_intent"]
if distinct_values > 1:
    columns_to_drop.append("person_home_ownership")

loanDF = loanDF.drop(*columns_to_drop)

# Show the updated DataFrame
loanDF.show(5)


# COMMAND ----------

from pyspark.sql.functions import when, col

# Step 1: Calculate the median for person_age in the existing DataFrame
median_age = loanDF.approxQuantile("person_age", [0.5], 0.0)[0]  # Median (50th percentile)
print(f"Median Age: {median_age}")

# Step 2: Replace extreme outliers with the median
# Define threshold: e.g., outliers > 100
loanDF = loanDF.withColumn(
    "person_age",
    when(col("person_age") > 100, median_age).otherwise(col("person_age"))
)

# Show the updated DataFrame
loanDF.show()

# COMMAND ----------

replaced_rows = loanDF.filter(col("person_age") == median_age)

# Show only the replaced rows
replaced_rows.show()

# COMMAND ----------

from pyspark.sql.functions import mean, stddev, min, max

# Summary statistics for person_income and loan_amnt
loanDF.select(
    mean("person_income").alias("avg_income"),
    stddev("person_income").alias("stddev_income"),
    min("person_income").alias("min_income"),
    max("person_income").alias("max_income")
).show()

# COMMAND ----------

loanDF.printSchema()

# COMMAND ----------

loanDF.show()

# COMMAND ----------

loanDF.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Analysis of Categorical Variable

# COMMAND ----------

from pyspark.sql.functions import count
import matplotlib.pyplot as plt

# Step 1: Aggregate data by loan_status
loan_status_distribution = loanDF.groupBy("loan_status").agg(count("*").alias("count"))

# Step 2: Convert the aggregated data to Pandas
loan_status_pandas = loan_status_distribution.toPandas()

# Step 3: Plot using Seaborn
plt.figure(figsize=(10, 6))
plt.bar(loan_status_pandas["loan_status"], loan_status_pandas["count"], color=["orange", "skyblue"])
plt.title("Loan Approval Count")
plt.xlabel("Loan Status (0: Rejected, 1: Approved)")
plt.ylabel("Count")
plt.xticks([0, 1], labels=["Rejected", "Approved"])
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ###Correlation with Loan Status - Heatmap

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# STEP 1: Get all columns with numeric types 
numerical_cols = [field.name for field in loanDF.schema.fields if field.dataType.simpleString() in ["int", "double"]]

#  Ensure all numerical columns are cast to double 
for col_name in numerical_cols:
    loanDF = loanDF.withColumn(col_name, col(col_name).cast("double"))

# STEP 2: Drop rows with nulls in selected numeric columns
loanDF_clean = loanDF.dropna(subset=numerical_cols)

# STEP 3: Assemble features
assembler = VectorAssembler(inputCols=numerical_cols, outputCol="features")
vector_df = assembler.transform(loanDF_clean).select("features")

# STEP 4: Compute correlation matrix
correlation_matrix = Correlation.corr(vector_df, "features").head()[0].toArray()

# STEP 5: Extract correlation with target variable
target_variable = "loan_status"
target_index = numerical_cols.index(target_variable)

# Create Pandas DataFrame of correlation values
target_corr = pd.DataFrame(
    correlation_matrix[:, target_index],
    index=numerical_cols,
    columns=[target_variable]
).sort_values(by=target_variable, ascending=False)

# STEP 6: Plot heatmap for correlations with target variable
plt.figure(figsize=(4, 6))
sns.heatmap(target_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title(f"Correlation with {target_variable}")
plt.show()


# COMMAND ----------

#Model Evaluation

# Split data into train and test sets
train_data, test_data = loanDF.randomSplit([0.7, 0.3], seed=42)
print(f"""There are {train_data.count()} rows in the training set,
and {test_data.count()} in the test set""")

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# Define input features that exist and are numeric
input_features = ["person_income", "loan_amnt", "cb_person_cred_hist_length", "loan_intent_index"]

# Assemble features
vecAssembler = VectorAssembler(inputCols=input_features, outputCol="features")
vecTrainDF = vecAssembler.transform(train_data)

# Show selected columns for confirmation
vecTrainDF.select(*input_features, "features", "loan_status").show(10)

# Train logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="loan_status")
lr_model = lr.fit(vecTrainDF)

# Optional: Evaluate model
predictions = lr_model.transform(vecTrainDF)
predictions.select("loan_status", "prediction", "probability").show(10)


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Define actual available numeric columns
feature_columns = [
    "person_age", "person_income", "person_emp_length", 
    "loan_amnt", "loan_int_rate", "loan_percent_income", 
    "cb_person_cred_hist_length", "cb_person_default_on_file", 
    "loan_intent_index"
]

# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
train_data = assembler.transform(train_data)

# Show result
train_data.select(*feature_columns, "features").show(5)


# COMMAND ----------

train_data.printSchema()

# COMMAND ----------

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
test_data = assembler.transform(test_data)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# Step 1: Define input features
feature_columns = [
    "person_age", "person_income", "person_emp_length", 
    "loan_amnt", "loan_int_rate", "loan_percent_income", 
    "cb_person_cred_hist_length", "cb_person_default_on_file", 
    "loan_intent_index"
]

# Step 2: Cast to double to avoid assembler errors
for col_name in feature_columns + ["loan_status"]:
    train_data = train_data.withColumn(col_name, col(col_name).cast("double"))

#  Step 3: Drop existing features column if it exists
if "features" in train_data.columns:
    train_data = train_data.drop("features")

# Step 4: Drop rows with nulls
train_data_clean = train_data.dropna(subset=feature_columns + ["loan_status"])

# Step 5: Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
train_data_vector = assembler.transform(train_data_clean)

# Step 6: Train logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="loan_status")
model = lr.fit(train_data_vector)

# Optional: Show model output
print("Coefficients:", model.coefficients)
print("Intercept:", model.intercept)


# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler

# Step 1: Define the same feature columns used in training
feature_columns = [
    "person_age", "person_income", "person_emp_length", 
    "loan_amnt", "loan_int_rate", "loan_percent_income", 
    "cb_person_cred_hist_length", "cb_person_default_on_file", 
    "loan_intent_index"
]

# Step 2: Cast all columns (including label if available) to double
for col_name in feature_columns:
    test_data = test_data.withColumn(col_name, col(col_name).cast("double"))

# Step 3: Drop rows with nulls
test_data_clean = test_data.dropna(subset=feature_columns)

# Step 4: Drop existing 'features' column if it exists
if "features" in test_data_clean.columns:
    test_data_clean = test_data_clean.drop("features")

# Step 5: Reuse the same VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
test_data_vector = assembler.transform(test_data_clean)

# Step 6: Make predictions
predictions = model.transform(test_data_vector)

# Step 7: Show results
predictions.select("loan_status", "prediction", "probability").show(10)


# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="loan_status", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")


# COMMAND ----------

# MAGIC %md
# MAGIC ###Confusion Matrix

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 3: Convert predictions to Pandas DataFrame
predictions_pandas = predictions.select("loan_status", "prediction").toPandas()

# Step 4: Generate classification report
print("Classification Report:")
print(classification_report(predictions_pandas["loan_status"], predictions_pandas["prediction"]))

# Step 5: Generate confusion matrix
cm = confusion_matrix(predictions_pandas["loan_status"], predictions_pandas["prediction"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected", "Approved"])

# Step 6: Visualize the confusion matrix
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

#  Step 1: Use existing valid numeric columns
selected_columns = [
    'person_income', 
    'loan_amnt', 
    'loan_int_rate', 
    'loan_percent_income', 
    'cb_person_default_on_file'  # <- corrected column name
]

# Step 2: Cast all columns to double (important!)
for col_name in selected_columns:
    loanDF = loanDF.withColumn(col_name, col(col_name).cast("double"))

# Step 3: Drop rows with nulls in selected columns
loanDF_clean = loanDF.dropna(subset=selected_columns)

# Step 4: Assemble features for clustering
vec_assembler = VectorAssembler(inputCols=selected_columns, outputCol="features")
df_vector = vec_assembler.transform(loanDF_clean)

# Step 5: Show result
df_vector.select(*selected_columns, "features").show(5)


# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col

# Step 1: Use correct and existing columns only
selected_columns = [
    'person_income', 
    'loan_amnt', 
    'loan_int_rate', 
    'loan_percent_income', 
    'cb_person_default_on_file'  # ← this is the corrected column name
]

# Step 2: Cast all to double and drop nulls
for col_name in selected_columns:
    loanDF = loanDF.withColumn(col_name, col(col_name).cast("double"))

loanDF_clean = loanDF.dropna(subset=selected_columns)

# Step 3: Assemble features
vec_assembler = VectorAssembler(inputCols=selected_columns, outputCol="features")
df_vector = vec_assembler.transform(loanDF_clean)

# Step 4: Run KMeans Clustering
kmeans = KMeans().setK(3).setSeed(1).setFeaturesCol("features").setPredictionCol("prediction")
model = kmeans.fit(df_vector)

# Step 5: Make predictions
predictions = model.transform(df_vector)

# Step 6: Show results (with correct columns)
predictions.select(
    'person_income', 
    'loan_amnt', 
    'loan_int_rate', 
    'loan_percent_income', 
    'cb_person_default_on_file', 
    'prediction'
).show(5)

# Step 7: Evaluate clustering (optional)
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette:.3f}")


# COMMAND ----------

# Evaluate clustering results using Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print(f"Silhouette with squared Euclidean distance = {silhouette}")


# COMMAND ----------

# MAGIC %md
# MAGIC ###Scatter Plot

# COMMAND ----------

# Convert to Pandas for plotting (ensure small enough to fit in memory)
pandas_df = predictions.select(
    'person_income', 
    'loan_amnt', 
    'loan_int_rate', 
    'loan_percent_income', 
    'cb_person_default_on_file',  
    'prediction'
).toPandas()

# Plot the clusters using matplotlib/seaborn
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='person_income', 
    y='loan_amnt', 
    hue='prediction', 
    data=pandas_df, 
    palette='Set1'
)
plt.title('Loan Applicants Clustering Based on Income and Loan Amount')
plt.xlabel('Income')
plt.ylabel('Loan Amount')
plt.grid(True)
plt.show()


# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

# Create Decision Tree model
dt = DecisionTreeClassifier(featuresCol="features", labelCol="loan_status")

# Fit the model
dt_model = dt.fit(train_data_vector)

# Make predictions
dt_predictions = dt_model.transform(test_data_vector)

# Show results
dt_predictions.select("loan_status", "prediction", "probability").show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Decision Tree Classifier

# COMMAND ----------


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


feature_columns = [
    "person_age", "person_income", "person_emp_length", "loan_grade",
    "loan_amnt", "loan_int_rate", "loan_status", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
    "person_home_ownership_index", "loan_intent_index"
]


loanDF_clean = loanDF.dropna(subset=feature_columns)


assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
final_data = assembler.transform(loanDF_clean)


train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=1234)


dt = DecisionTreeClassifier(labelCol="loan_status", featuresCol="features", maxDepth=5)
dt_model = dt.fit(train_data)


dt_predictions = dt_model.transform(test_data)


evaluator = MulticlassClassificationEvaluator(
    labelCol="loan_status", predictionCol="prediction", metricName="accuracy"
)
dt_accuracy = evaluator.evaluate(dt_predictions)
print(f" Decision Tree Test Accuracy: {dt_accuracy:.4f}")


print("\nDecision Tree Model Structure:\n")
print(dt_model.toDebugString)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Decision Tree visualization

# COMMAND ----------


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


train_pd = train_data.select(
    "person_age", "person_income", "person_emp_length", "loan_grade",
    "loan_amnt", "loan_int_rate", "loan_status", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
    "person_home_ownership_index", "loan_intent_index"
).toPandas()


X = train_pd.drop("loan_status", axis=1)
y = train_pd["loan_status"]


dt_sklearn = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_sklearn.fit(X, y)


plt.figure(figsize=(20, 10))
plot_tree(
    dt_sklearn, 
    feature_names=X.columns.tolist(),   
    class_names=["Rejected", "Approved"],
    filled=True, 
    rounded=True, 
    fontsize=12
)
plt.title("Decision Tree Visualization")
plt.show()


# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# F1 Score
dt_f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="loan_status", 
    predictionCol="prediction", 
    metricName="f1"
)
dt_f1 = dt_f1_evaluator.evaluate(dt_predictions)
print(f"F1 Score: {dt_f1:.4f}")

# Precision (Weighted)
dt_precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="loan_status", 
    predictionCol="prediction", 
    metricName="weightedPrecision"
)
dt_precision = dt_precision_evaluator.evaluate(dt_predictions)
print(f"Precision (Weighted): {dt_precision:.4f}")

# Recall (Weighted)
dt_recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="loan_status", 
    predictionCol="prediction", 
    metricName="weightedRecall"
)
dt_recall = dt_recall_evaluator.evaluate(dt_predictions)
print(f"Recall (Weighted): {dt_recall:.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC Logistic Regression Code (with Coefficient Bar Chart)

# COMMAND ----------

# Step 0: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

# Step 1: Define the correct feature columns (exclude label)
all_cols = final_data.columns
feature_cols = [col for col in all_cols if col not in ['loan_status', 'Loan_Status_Index', 'features']]

# Step 2: Assemble features into a new column to avoid conflict
assembler = VectorAssembler(inputCols=feature_cols, outputCol="new_features")
assembled_data = assembler.transform(final_data)

# Step 3: Split the data
train_data, test_data = assembled_data.randomSplit([0.8, 0.2], seed=1234)

# Step 4: Train Logistic Regression model
lr = LogisticRegression(labelCol="loan_status", featuresCol="new_features")
lr_model = lr.fit(train_data)

# Step 5: Predict on test data
lr_predictions = lr_model.transform(test_data)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# F1 Score
lr_f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="loan_status",
    predictionCol="prediction",
    metricName="f1"
)
lr_f1 = lr_f1_evaluator.evaluate(lr_predictions)
print(f"F1 Score: {lr_f1:.4f}")

# Precision (Weighted)
lr_precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="loan_status",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
lr_precision = lr_precision_evaluator.evaluate(lr_predictions)
print(f"Precision (Weighted): {lr_precision:.4f}")

# Recall (Weighted)
lr_recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="loan_status",
    predictionCol="prediction",
    metricName="weightedRecall"
)
lr_recall = lr_recall_evaluator.evaluate(lr_predictions)
print(f"Recall (Weighted): {lr_recall:.4f}")


# Step 6: Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="loan_status",
    predictionCol="prediction",
    metricName="accuracy"
)
lr_accuracy = evaluator.evaluate(lr_predictions)
print(f" Logistic Regression Test Accuracy: {lr_accuracy:.4f}")


# COMMAND ----------

# Step 0: Extract coefficients from trained Logistic Regression model
coefficients = lr_model.coefficients.toArray()

# Step 1: Get the actual input feature names used by the assembler
feature_names = assembler.getInputCols()

# Step 2: Ensure Python's min() is used (optional but safe)
from builtins import min

# Step 3: Align lengths
min_len = min(len(coefficients), len(feature_names))
feature_names = feature_names[:min_len]
coefficients = coefficients[:min_len]

# Step 4: Create the DataFrame
import pandas as pd
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})



# COMMAND ----------



#  Plot the bar chart
plt.figure(figsize=(14, 8))
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis', edgecolor='black')

# Annotate bars with coefficient values
for index, value in enumerate(coef_df['Coefficient']):
    plt.text(value, index, f'{value:.4f}', color='black', va='center', fontweight='bold')

plt.title('Logistic Regression Feature Coefficients (Corrected)', fontsize=18)
plt.xlabel('Coefficient Value', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.axvline(0, color='black', linestyle='--')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# F1 Score
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="loan_status", 
    predictionCol="prediction", 
    metricName="f1"
)
f1_score = f1_evaluator.evaluate(lr_predictions)
print(f"F1 Score: {f1_score:.4f}")

# Precision
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="loan_status", 
    predictionCol="prediction", 
    metricName="weightedPrecision"
)
precision = precision_evaluator.evaluate(lr_predictions)
print(f"Precision (Weighted): {precision:.4f}")

# Recall
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="loan_status", 
    predictionCol="prediction", 
    metricName="weightedRecall"
)
recall = recall_evaluator.evaluate(lr_predictions)
print(f"Recall (Weighted): {recall:.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC ###Probability Distribution for Loan Approval

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Extract predicted probabilities for class 1 (loan approved)
probs = lr_predictions.select('probability').rdd.map(lambda row: row['probability'][1]).collect()
probs = np.array(probs)

# Step 2: Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(probs, bins=30, edgecolor='black', color='skyblue')
plt.title('Probability Distribution for Loan Approval (Class 1)', fontsize=16)
plt.xlabel('Predicted Probability of Approval', fontsize=14)
plt.ylabel('Number of Applicants', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ###Histograms

# COMMAND ----------

# Convert Spark DataFrame to Pandas DataFrame
pdf = final_data.select(
    'person_income',
    'loan_amnt',
    'person_age',
    'loan_int_rate',
    'cb_person_cred_hist_length'
).toPandas()


# Relevant numerical columns in credit risk prediction
numerical_columns = [
    'person_income', 
    'loan_amnt', 
    'person_age', 
    'loan_int_rate', 
    'cb_person_cred_hist_length'
]

titles = [
    'Income Distribution', 
    'Loan Amount Distribution', 
    'Age Distribution', 
    'Interest Rate Distribution', 
    'Credit History Length Distribution'
]

# Plot histograms
fig, axs = plt.subplots(3, 2, figsize=(14, 10))
axs = axs.ravel()

for i, col in enumerate(numerical_columns):
    axs[i].hist(pdf[col].dropna(), bins=30, color='skyblue', edgecolor='black')
    axs[i].set_title(titles[i])
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('Count')

# Remove the empty sixth subplot
fig.delaxes(axs[5])

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ###Correlation Analysis

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Convert your Spark DataFrame to Pandas (if not already)
pdf = loanDF.select(
    'person_income', 
    'loan_amnt', 
    'person_age', 
    'loan_int_rate', 
    'cb_person_cred_hist_length',
    'loan_status'  # Ensure loan_status is numeric (0 or 1)
).toPandas()

# If loan_status is string (e.g., "Y"/"N"), convert it to numeric
pdf['loan_status'] = pdf['loan_status'].map({'Y': 1, 'N': 0})

# Compute correlation matrix
corr_matrix = pdf.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f", square=True)
plt.title('Correlation Matrix of Credit Risk Variables')
plt.show()
