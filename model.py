import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# import matplotlib
import streamlit as st

# Set the default figure size for plots
matplotlib.rcParams["figure.figsize"] = (20, 10)

# Load the dataset
df1 = pd.read_csv("Bengaluru_House_Data.csv")

# Display the shape and initial counts by area type
print(df1.shape)
print(df1.groupby('area_type')['area_type'].count())

# Drop unwanted columns
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')

# Drop rows with missing values
df3 = df2.dropna()

# Extract BHK (number of bedrooms) from 'size' column
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3['bhk'].unique())

# Define a function to check if a value can be converted to a float
def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

# Identify non-float 'total_sqft' entries
print(df3[~df3['total_sqft'].apply(is_float)].head(10))

# Define a function to convert 'total_sqft' to a numeric value
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

# Apply conversion to 'total_sqft' and filter out rows with invalid values
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4 = df4[df4['total_sqft'].notnull()]

# Calculate price per square foot
df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']

# Clean and standardize the 'location' column
df5['location'] = df5['location'].apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)

# Label locations with less than or equal to 10 occurrences as 'other'
location_stats_less_than_10 = location_stats[location_stats <= 10]
df5['location'] = df5['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

# Remove outliers based on total square feet per BHK
df6 = df5[~(df5['total_sqft'] / df5['bhk'] < 300)]

# Function to remove outliers based on price per square foot
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf['price_per_sqft'])
        st = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)

# Function to plot scatter chart for price vs. total square feet
def plot_scatter_chart(df, location):
    bhk2 = df[(df['location'] == location) & (df['bhk'] == 2)]
    bhk3 = df[(df['location'] == location) & (df['bhk'] == 3)]
    plt.figure(figsize=(15, 10))
    plt.scatter(bhk2['total_sqft'], bhk2['price'], color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3['total_sqft'], bhk3['price'], marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()

plot_scatter_chart(df7, "Rajaji Nagar")
plot_scatter_chart(df7, "Hebbal")

# Function to remove BHK outliers
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['price_per_sqft']),
                'std': np.std(bhk_df['price_per_sqft']),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df['price_per_sqft'] < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

df8 = remove_bhk_outliers(df7)

# Plot histograms to visualize data
plt.hist(df8['price_per_sqft'], rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()

plt.hist(df8['bath'], rwidth=0.8)
plt.xlabel("Number of Bathrooms")
plt.ylabel("Count")
plt.show()

# Remove properties with unrealistic number of bathrooms
df9 = df8[df8['bath'] < df8['bhk'] + 2]

# Prepare data for model training
df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')
dummies = pd.get_dummies(df10['location'])
df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')
df12 = df11.drop('location', axis='columns')

# Define X and y for the model
X = df12.drop(['price'], axis='columns')
y = df12['price']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train a Linear Regression model
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

# Evaluate model performance
print(lr_clf.score(X_test, y_test))

# Cross-validation
from sklearn.model_selection import ShuffleSplit, cross_val_score
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
print(cross_val_score(LinearRegression(), X, y, cv=cv))

# Function to predict prices
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return lr_clf.predict([x])[0]

# Example predictions
print(predict_price('1st Phase JP Nagar', 1000, 2, 2))
print(predict_price('Indira Nagar', 1000, 2, 2))

# Save model using pickle
import pickle
with open('bangalore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(lr_clf, f)

# Save columns to a JSON file
import json
columns = {'data_columns': [col.lower() for col in X.columns]}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))
