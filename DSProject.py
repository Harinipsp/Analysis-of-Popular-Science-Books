# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('C:/Users/harin/Downloads/archive/final_book_dataset_kaggle2.csv')
Amazon_books = pd.DataFrame(data)
print(Amazon_books.head())
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth',100)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(Amazon_books.columns)
print(Amazon_books.shape)
print(Amazon_books.isna().sum())
Amazon_books.dropna(subset='n_reviews',inplace=True)
import matplotlib.pyplot as plt
plt.hist(Amazon_books['price'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.show()
median = Amazon_books['price'].median()
Amazon_books['price'].fillna(median, inplace=True)
Amazon_books.isna().sum()
Amazon_books.info()
Amazon_books['n_reviews'] = Amazon_books['n_reviews'].str.replace(',', '').astype(float)
Amazon_books['avg_reviews'] = Amazon_books['avg_reviews'].astype(float)
non_numeric_rows = Amazon_books[Amazon_books['pages'].str.replace('.', '', 1).str.isnumeric() == False]
Amazon_books = Amazon_books.drop(non_numeric_rows.index)
Amazon_books['pages'] = Amazon_books['pages'].astype(float)
import matplotlib.pyplot as plt
plt.hist(Amazon_books['price'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Books by prices')
plt.show()
print(Amazon_books['price'].mean())
import matplotlib.pyplot as plt
plt.scatter(Amazon_books['avg_reviews'],Amazon_books['price'])
plt.ylabel('Price')
plt.xlabel('Average Reviews')
plt.title('Scatter Plot of Price vs. Average Reviews')
coef = np.polyfit(Amazon_books['avg_reviews'],Amazon_books['price'],1)
trendline = np.poly1d(coef)
plt.plot(Amazon_books['avg_reviews'],trendline(Amazon_books['avg_reviews']),"r--")
plt.show()
# Scatter plot
plt.scatter(Amazon_books['pages'], Amazon_books['price'])
plt.ylabel('price')
plt.xlabel('Number of pages')
plt.title('Scatter Plot of Number of pages vs. price')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt  # Import matplotlib for customization

# Calculate the correlation matrix
correlation_matrix = Amazon_books[['pages', 'price']].corr()

# Create a heatmap with labels
plt.figure(figsize=(8, 6))  # Set the figure size
sns.heatmap(
    correlation_matrix,
    annot=True,                     
    fmt=".2f",               
    cbar=True,               
    square=True,                      
)
plt.title('Correlation Heatmap')  # Add a title to the plot
plt.show()

