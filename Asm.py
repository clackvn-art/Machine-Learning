import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score

#Giai đoạn 1
# Load the California housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

def discover():
    # Display basic information about the dataset
    print("Dataset Information:")
    print(df.info())
    print("\nDataset shape:")
    print(df.shape)
    print("\nDataset description:")
    print(df.describe())
discover()




# Split the dataset into features and target variable
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for missing values
print("\nMissing values in each column:")
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.isnull().sum())

train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

print("\nNaN sau khi xu ly:")
print("Train:", X_train.isnull().sum().sum())
print("Test :", X_test.isnull().sum().sum())




# Remove outliers based on the IQR and clipping method
Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1

lower_train = Q1 - 1.5 * IQR
upper_train = Q3 + 1.5 * IQR

X_train = X_train.clip(
lower=lower_train,
upper=upper_train,
axis=1
)

X_test = X_test.clip(
lower=lower_train,
upper=upper_train,
axis=1
)


# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")


tolerance = 0.5
y_test_class = (y_test > tolerance).astype(int)
y_pred_class = (y_pred > tolerance).astype(int)

# Calculate accuracy
print(f"Accuracy: {accuracy_score(y_test_class, y_pred_class)}")

# Create a confusion matrix
cm = confusion_matrix(y_test_class, y_pred_class)
print("\nConfusion Matrix:")
print(cm)


# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted <= 0.5', 'Predicted > 0.5'], yticklabels=['Actual <= 0.5', 'Actual > 0.5'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Regression')
plt.show()
