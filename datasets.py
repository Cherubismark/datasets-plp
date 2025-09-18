
# Task 1: Load and Explore Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset from sklearn
iris = load_iris(as_frame=True)
df = iris.frame

# Display first rows
print("First 5 rows:")
print(df.head())

# Check data types and missing values
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# No missing values in Iris dataset. If there were, handle them like this:
# df = df.dropna()   # drop missing rows
# df = df.fillna(value=0)   # fill missing with 0



# Task 2: Basic Data Analysis

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Group by species and compute mean for numerical columns
grouped = df.groupby("target").mean()
print("\nMean values grouped by species:")
print(grouped)

# Observations
# Example: Check which species has highest average petal length
max_species = grouped["petal length (cm)"].idxmax()
print(f"\nSpecies with highest average petal length: {iris.target_names[max_species]}")


# =============================
# Task 3: Data Visualization
# =============================

# Line chart - show sepal length trend over rows (acts like time)
plt.figure(figsize=(8,5))
plt.plot(df["sepal length (cm)"])
plt.title("Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.show()

# Bar chart - average petal length per species
plt.figure(figsize=(8,5))
sns.barplot(x="target", y="petal length (cm)", data=df, estimator="mean")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.xticks([0,1,2], iris.target_names)
plt.show()

# Histogram - distribution of sepal width
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=15, edgecolor="black")
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot - sepal length vs petal length
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="target", data=df, palette="Set1")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species", labels=iris.target_names)
plt.show()


# =============================
# Findings
# =============================

# 1. Species differ clearly by petal length and width.
# 2. Sepal width distribution is slightly skewed.
# 3. Line trend is arbitrary since dataset has no time index, but shows variance across samples.
# 4. Scatter plot shows Iris Setosa has small petals, while Virginica has the largest.
