import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 1 and 2: Read the dataset
df = pd.read_csv("data.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

# Step 2: Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Drop missing values
df = df.dropna(axis=1, how="all")
df = df.dropna()

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Step 3: Drop unnecessary columns
# The id column is not useful for analysis
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Some versions of this dataset have an empty column called Unnamed: 32
if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])

print("\nColumns after dropping unnecessary columns:")
print(df.columns)

# Step 4: Encode diagnosis column
# M = malignant, B = benign
label_encoder = LabelEncoder()
df["diagnosis_encoded"] = label_encoder.fit_transform(df["diagnosis"])

print("\nDiagnosis counts:")
print(df["diagnosis"].value_counts())

print("\nEncoded diagnosis values:")
print(df[["diagnosis", "diagnosis_encoded"]].head())

# Step 4: Choose numerical features
features = df.drop(columns=["diagnosis", "diagnosis_encoded"])

# Step 4: Scaling / normalization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

scaled_df = pd.DataFrame(
    scaled_features,
    columns=features.columns
)

scaled_df["diagnosis"] = df["diagnosis"].values
scaled_df["diagnosis_encoded"] = df["diagnosis_encoded"].values

# Save the preprocessed dataset
scaled_df.to_csv("data_refined.csv", index=False)

print("\nPreprocessed dataset saved as data_refined.csv")
print(scaled_df.head())

# Step 5: Pair plot for selected features
selected_features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "diagnosis"
]

sns.pairplot(df[selected_features], hue="diagnosis")
plt.suptitle("Pair Plot of Selected Breast Cancer Features", y=1.02)
plt.show()

# Step 5: Correlation matrix heatmap
plt.figure(figsize=(14, 10))
correlation_matrix = df.drop(columns=["diagnosis"]).corr()
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Step 5: Box plots for selected features
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]])
plt.title("Box Plot of Selected Features")
plt.xticks(rotation=45)
plt.show()

# Step 6: Violin plots
plt.figure(figsize=(12, 6))
sns.violinplot(data=df[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]])
plt.title("Violin Plot of Selected Features")
plt.xticks(rotation=45)
plt.show()

print("""
A violin plot shows the distribution of numerical data.
It combines a box plot with a density plot, so it shows both the spread of values and where most values are concentrated.

Based on the violin plots and box plots, some features appear to have outliers.
For example, area_mean and perimeter_mean may show extreme values compared with the majority of the data.
These outliers can be important because they may represent unusual tumor measurements.
""")
