import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
df = pd.read_csv('car_price_prediction.csv')

#print(df.head())

df1 = df.drop(['Doors'], axis= 'columns')

#print(df1.isnull().sum())

df1['Levy'] = df1['Levy'].replace('-', np.nan)  # Replace with NaN
df1['Levy'] = df1['Levy'].astype(float)

df1['Levy'].fillna(df1['Levy'].median(), inplace=True)

#print(df1.head(20))

#print(df1['Manufacturer'].nunique())
#print(df1['Category'].nunique())     # 11
#print(df1['Fuel type'].nunique())    # 7
#print(df1['Gear box type'].nunique())    # 4
#print(df1['Drive wheels'].nunique())    # 3
#print(df1['Wheel'].nunique())    # 2
#print(df1['Color'].nunique())    # 16

#print(df1.groupby('Wheel')['Price'].mean())    # Checking if wheel direction has impact on price


df2 = pd.get_dummies(df1, columns=['Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color'], drop_first=True, dtype=int)
#print(df2.columns)

#print(df2[df2.duplicated(keep=False)])   
df3 = df2.drop_duplicates()

# Clean Mileage (remove ' km' and commas)
df3['Mileage'] = df3['Mileage'].str.replace(' km', '', regex=False).str.replace(',', '').astype(float)

# Clean Engine volume (remove ' Turbo' text if present)
df3['Engine volume'] = df3['Engine volume'].str.replace(' Turbo', '', regex=False).astype(float)

# Ensure Levy is numeric
df3['Levy'] = pd.to_numeric(df3['Levy'], errors='coerce')
print(df3.shape[1])
""" 

num_col = ['Price', 'Levy', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags']
outliers = {}
for col in num_col:
    Q1 = df3[col].quantile(0.25)
    Q3 = df3[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect outliers
    outlier_rows = df3[(df3[col] < lower_bound) | (df3[col] > upper_bound)]

    outliers[col] = outlier_rows.shape[0]   # number of outliers in that column

  #  print(f"{col}:")
 #   print(f"  Q1 = {Q1}, Q3 = {Q3}, IQR = {IQR}")
 #   print(f"  Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")
#    print(f"  Outliers detected = {outlier_rows.shape[0]}\n")
 """
#df3["Price"].hist(bins=50)
#plt.xlabel("Price")
#plt.ylabel("Count")
#plt.title("Car Price Distribution")
#plt.show()

# Check how many cars > 6M
#print("Cars:", (df3["Price"] <= 100).sum())
#print("Total cars:", len(df3))

df4 = df3[df3["Price"] <= 100000]
df5 = df4[df4['Price'] >= 100]
df6 = df5[df5['Engine volume'] != 0]
df7 = df6[df6['Mileage'] <= 999999]

#count_bad_cars = ((df7['Prod. year'] < 2010) & (df7['Mileage'] == 0)).sum()
#print("Number of such cars:", count_bad_cars)


df7 = df7[~((df7['Prod. year'] < 2010) & (df7['Mileage'] == 0))]
""" 
# Number of unique manufacturers
total_manufacturers = df7['Manufacturer'].nunique()
print("Total unique manufacturers:", total_manufacturers)

# Number of unique models
total_models = df7['Model'].nunique()
print("Total unique models:", total_models)

# Number of unique manufacturer + model combinations
total_combinations = df7[['Manufacturer', 'Model']].drop_duplicates().shape[0]
print("Total unique manufacturer-model combinations:", total_combinations)


df7['Manufacturer'] = df7['Manufacturer'].str.strip().str.lower()
dup_models = df7.groupby('Model')['Manufacturer'].nunique()
dup_models = dup_models[dup_models > 1]
print(dup_models)
df7['Model'] = df7['Model'].str.strip().str.lower()


# Number of unique manufacturers
total_manufacturers = df7['Manufacturer'].nunique()
print("Total unique manufacturers:", total_manufacturers)

# Number of unique models
total_models = df7['Model'].nunique()
print("Total unique models:", total_models)

# Number of unique manufacturer + model combinations
total_combinations = df7[['Manufacturer', 'Model']].drop_duplicates().shape[0]
print("Total unique manufacturer-model combinations:", total_combinations) """


#print(df7.duplicated().sum())
#print(df7['Car Name'].nunique())

# Example: using df7 and 'Car Name'
#threshold = 0.01  # 1%

# Calculate frequency percentages
#freq = df7['Car Name'].value_counts(normalize=True)

# Find categories below threshold
#rare_cars = freq[freq < threshold].index

# Replace them with "Others"
#df7['Car Name'] = df7['Car Name'].replace(rare_cars, 'Others')

# Check result
#print(df7['Car Name'].value_counts(normalize=True).head(20))
#print(df7['Car Name'].nunique())

#df8 = pd.get_dummies(df7, 'Car Name', drop_first=True)
#print(df8.shape[1])

# Number of columns
#print("Total number of columns:", len(df8.columns))
#with open("columns_list.txt", "w", encoding="utf-8") as f:
 #   for col in df8.columns:
  #      f.write(col + "\n")

# Check if 'Manufacturer' exists
print("Columns in df7:", df7.columns)

# Number of unique manufacturers
print("Total unique manufacturers:", df7['Manufacturer'].nunique())

# One-hot encoding
df8 = pd.get_dummies(df7, columns=['Manufacturer'], drop_first=True)
print("Total columns after encoding:", df8.shape[1])

#with open("new_columns_list.txt", "w", encoding="utf-8") as f:
    #for col in df8.columns:
        #f.write(col + "\n")

df8 = df8.drop(['Manufacturer_სხვა'], axis= 'columns')     # problematic
print("Columns in df8:", df8.columns)

X = df8.drop(['ID', 'Price'], axis=1)     # Features
Y = df8['Price']     # Target

joblib.dump(X, 'X.pkl')
joblib.dump(Y, 'Y.pkl')