import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def bai_1 ():
    np.random.seed(40) #cố định số ngẫu nhiên để chạy lại ra cùng kết quả
    X=np.arange(1,11).reshape(-1,1) #tạo 10 giá trị X, đổi thành dạng cột
    y=2*X.ravel()+1+np.random.randn(10) #tạo y theo công thức, thêm nhiễu ngẫu nhiên
    model=LinearRegression().fit(X,y)#tạo huấn luyện mô hình hồi quy
    print("He số, hệ số chặn: ",model.coef_[0],model.intercept_) #in ra hệ số góc và hệ số chặn
bai_1 ()

def bai_2_3():
    data=load_diabetes() #load dữ liệu
    X=data.data[:, 2].reshape(-1, 1) #lấy dữ liệu đầu vào
    y=data.target #lấy dữ liệu đầu ra

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) #chia dữ liệu thành tập huấn luyện và tập kiểm tra
    
    scaler=StandardScaler() #chuẩn hóa dữ liệu
    X_train=scaler.fit_transform(X_train) #chuẩn hóa tập huấn luyện
    X_test=scaler.transform(X_test) #chuẩn hóa tập kiểm tra

    model=LinearRegression().fit(X_train,y_train) #huấn luyện mô hình hồi
    y_pred=model.predict(X_test) #dự đoán dữ liệu kiểm tra

    # MAE: Sai số tuyệt đối trung bình.
    # Giá trị càng nhỏ thì dự đoán gia trị càng gần với thực tế.
    mae = mean_absolute_error(y_test, y_pred)

    # MSE: Sai số bình phương trung bình.
    # Phát hiện các số lớn bằng các bình phương sai số.
    # Giá trị càng nhỏ thì mô hình càng tốt.
    mse = mean_squared_error(y_test, y_pred)

    # R-squared: Hệ số xác định
    # Cho biết mô hình giải thích được bao nhiêu biến thiên của y.
    # Gần 1: mô hình giải thích tốt.
    # Gần 0: khả năng giải thích yếu.
    # Âm: mô hình tệ hơn việc dự đoán bằng giá trị trung bình.
    r2 = r2_score(y_test, y_pred)

    print("MAE: ",mae) 
    print("MSE: ",mse) 
    print("R2: ",r2)  

    # Bài 3
    plt.scatter(X_test,y_test)
    #vẽ các điểm dự đoán
    plt.plot(X_test,y_pred,"r")
    #đặt tên trục ngang
    plt.xlabel("BMI")
    #đặt tên trục dọc
    plt.ylabel("Diabetes target")
    #hiển thị biểu đồ
    plt.show()
bai_2_3()

