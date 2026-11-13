import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def bai_1 ():
    data = {
    "Tuổi": [22,25,np.nan,29,45,150],
    "Điểm_Thi": [8.0,np.nan,6.5,9.0,5.5,4.0]
    }
    df=pd.DataFrame(data)
    #1.1 Thong ke tổng số lượng giá trị bị thiếu (NaN) trên từng cột
    print("1. Số lượng giá trị thiếu (NaN) trên từng cột:\n",df.isnull().sum())
    #1.2 ĐIền giá trị thiếu ở cột ĐIểm_Thi bằng giá trị trung vị (Median)
    median_diem = df["Điểm_Thi"].median()
    df["Điểm_Thi"]=df["Điểm_Thi"].fillna(median_diem)
    #1.3 Xử lý giá trị ngoại lệ ở cột Tuổi (loại bỏ dòng chứa tuối >100)
    df=df[df["Tuổi"]<=100]
    print("\nDataFrame sau khi làm sạch dữ liệu:\n",df)
bai_1 ()

def bai_2 ():
    data = np.array([
        [20, 10],
        [25, 20],
        [30, 30],
        [40, 50],
        [50, 70],
        [60, 100]
    ])

    print("=== DU LIEU BAN DAU ===")
    print(data)

    # 2. Su dung StandardScaler
    standard_scaler = StandardScaler()

    data_standard = standard_scaler.fit_transform(data)

    print("\n=== SAU KHI SU DUNG STANDARDSCALER ===")
    print(data_standard)

    # 3. Su dung MinMaxScaler
    minmax_scaler = MinMaxScaler()

    data_minmax = minmax_scaler.fit_transform(data)

    print("\n=== SAU KHI SU DUNG MINMAXSCALER ===")
    print(data_minmax)
bai_2 ()


def bai_3 ():
    diabetes = load_diabetes()

    X = diabetes.data
    y = diabetes.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    print("Kich thước tạo X_train: ",X_train.shape)
    print("Kich thước tạo X_test: ",X_test.shape)
    print("Kich thước tạo y_train: ",y_train.shape)
    print("Kich thước tạo y_test: ",y_test.shape)
bai_3()

