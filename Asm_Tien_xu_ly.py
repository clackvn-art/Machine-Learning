import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Dữ liệu trước khi tiền xử lý
print("\nDữ liệu trước khi tiền xử lý:")
print("Train:", X_train.shape[0])
print("Test :", X_test.shape[0])

# Kiểm tra giá trị thiếu
print("\nMissing values in each column:")
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.isnull().sum())

train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

print("\nNaN sau khi xu ly:")
print("Train:", X_train.isnull().sum().sum())
print("Test :", X_test.isnull().sum().sum())

# Xử lý ngoại lệ dựa trên IQR và phương pháp clipping
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

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Dữ liệu sau tiền xử lý
print("\nDữ liệu sau tiền xử lý:")
print("Train:", X_train_scaled.shape[0])
print("Test :", X_test_scaled.shape[0])