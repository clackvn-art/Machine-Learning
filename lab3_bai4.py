from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


cali=fetch_california_housing()
X=cali.data
y=cali.target

#Chia dữ liệu theo tỷ lệ 80:20, cố định random_state
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler() #chuẩn hóa dữ liệu
X_train=scaler.fit_transform(X_train) #chuẩn hóa tập huấn luyện
X_test=scaler.transform(X_test) #chuẩn hóa tập kiểm tra

model=LinearRegression().fit(X_train,y_train) #huấn luyện mô hình hồi quy
y_pred=model.predict(X_test) #dự đoán dữ liệu kiểm tra

mse = mean_squared_error(y_test, y_pred)

#Visualize một phần kết quả.
plt.scatter(y_test[:100],y_pred[:100])
plt.xlabel("Giá trị thực tế")

plt.ylabel("Giá trị dự đoán")
plt.title("Dự đoán giá nhà tại California")
plt.show()