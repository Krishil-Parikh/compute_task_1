import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('/Users/krishilparikh/CODING/Compute/compute_task_1/listings_data (1).csv')
df.drop(['listing_name' , 'host_id' , 'host_name' , 'listing_id'] , axis='columns' , inplace=True)

print(df.head())

print(df.info())

print(df.isnull().sum())

# Handling missing values
df['reviews_per_month'].fillna(df['reviews_per_month'].median(), inplace=True)
df['last_review_date'].fillna(df['last_review_date'].mode()[0], inplace=True)

# Check for duplicates
print(f"Number of duplicate rows: {df.duplicated().sum()}")

# Remove duplicates if any
df.drop_duplicates(inplace=True)

# Correcting data types
df['last_review_date'] = pd.to_datetime(df['last_review_date'], format='%d-%m-%Y')

# Descriptive statistics
print(df.describe())

# Step 3: One-Hot Encoding for Categorical Variables
area_column = df[['area', 'borough']]
encoder = OneHotEncoder(sparse_output=False) 
area_encoded = encoder.fit_transform(area_column)
area_encoded_df = pd.DataFrame(area_encoded, columns=encoder.get_feature_names_out(['area', 'borough']))
df_encoded = pd.concat([df, area_encoded_df], axis=1)
df_encoded.drop(['area', 'borough'], axis='columns', inplace=True)

# Display the first few rows of the modified DataFrame
print(df_encoded.head())

# Plotting histograms for numeric variables like price, minimum_stay, no_of_reviews, reviews_per_month
plt.figure(figsize=(10, 6))
df['price (in dollars)'].hist(bins=30, edgecolor='black')
plt.title('Distribution of Prices')
plt.xlabel('Price (in dollars)')
plt.ylabel('Frequency')
plt.xlim(0,2000)
plt.show()

plt.figure(figsize=(10, 6))
df['minimum_stay'].hist(bins=30, edgecolor='black')
plt.title('Distribution of Minimum Stay')
plt.xlabel('Minimum Stay (nights)')
plt.ylabel('Frequency')
plt.xlim(0,400)
plt.show()

plt.figure(figsize=(10, 6))
df['no_of_reviews'].hist(bins=30, edgecolor='black')
plt.title('Distribution of Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
plt.xlim(0,400)
plt.show()

# Scatter plot between price and number of reviews
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price (in dollars)', y='no_of_reviews', data=df_encoded)
plt.title('Price vs Number of Reviews')
plt.xlabel('Price (in dollars)')
plt.ylabel('Number of Reviews')
plt.show()

# Heatmap for correlations
plt.figure(figsize=(12, 8))

# Select only numeric columns for correlation matrix
numeric_df = df.drop(['borough' , 'area'] , axis='columns').select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Boxplot for Price per Borough
plt.figure(figsize=(12, 8))
sns.boxplot(x='borough', y='price (in dollars)', data=df)
plt.title('Price Distribution per Borough')
plt.xlabel('Borough')
plt.ylabel('Price (in dollars)')
plt.show()

# Room Type Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='room_type', data=df_encoded)
plt.title('Room Type Distribution')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.show()

# Price per Room Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='room_type', y='price (in dollars)', data=df_encoded)
plt.title('Price Distribution per Room Type')
plt.xlabel('Room Type')
plt.ylabel('Price (in dollars)')
plt.show()

# Availability of Listings
plt.figure(figsize=(10, 6))
df['booking_availability'].hist(bins=30, edgecolor='black')
plt.title('Booking Availability Distribution')
plt.xlabel('Availability (days)')
plt.ylabel('Frequency')
plt.show()
