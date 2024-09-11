pip install numpy pandas scikit-learn matplotlib seaborn
import pandas as pd

# Load the training dataset
train_data = pd.read_csv('train.csv')

# View the first few rows of the dataset
print(train_data.head())
from google.colab import files
uploaded = files.upload()  # This will open a dialog box to upload files
from google.colab import files
uploaded = files.upload()  # This will open a dialog box to upload files
# Drop columns with too many missing values
train_data = train_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

# Fill missing values for numerical columns with the median
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())

# Fill missing values for categorical columns with the most frequent value
train_data['Electrical'] = train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])
# One-Hot Encoding for categorical variables
train_data = pd.get_dummies(train_data)
# Define the features and target
X = train_data.drop('SalePrice', axis=1)  # Features
y = train_data['SalePrice']  # Target
from sklearn.model_selection import train_test_split

# Split the dataset (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Impute missing values using SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median') # Use median imputation
X_train = imputer.fit_transform(X_train)

# Initialize the linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)
# Impute missing values in X_test using the same imputer
X_test = imputer.transform(X_test)

# Make predictions on the test set
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate R² Score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")