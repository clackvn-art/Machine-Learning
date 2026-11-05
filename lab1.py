import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris = datasets.load_iris()

def bai1 ():
    #1. Tạo bảng numpy 1 chiều chứa các số từ 1 đến 10
    arr = np.arange(1,11)
    print ('Mảng Numpy:', arr)
    #2. Tạo dataframe từ pandas chứa thôgn tin 5 sinh viên
    data = {
        'Họ tên': ['An', 'Bình', 'Dũng', 'Hà'],
        'Tuổi': ['20', '22', '21', '19'],
        'Điểm toán': [8, 5, 7, 9]
    }
    df = pd.DataFrame(data)

    #3. Hiển thị thông tin tổng quan và tính điểm trung bình môn toán
    print ("Thông tin tổng quan")
    print (df.info)

    #4. viết cho tôi hàm tính điểm trung bình toán
    avg_math = df['Điểm toán'].mean()
    print ("Điểm trung bình môn toán là:", avg_math)
# bai1()


# bai 2

# 1. Dự đoán giá nhà dựa trên diện tích
# Supervised Learning - Học có giám sát

# 2. Gom nhóm khách hàng dựa trên hành vi mua sắm
# Unsupervised Learning - Học không giám sát

# 3. Phan loai email la Spam hay khong
# Supervised Learning - Học có giám sát

# 4. Nhan dien chu viet tay tu hinh anh
# Supervised Learning - Học có giám sát

def bai3():
    #tải dữ liệu mẫu
    digits= datasets.load_digits()
    print ("Hình dạng dữ liệu (Số lượng mẫu, số đặc trưng):", digits.data.shape)
    print ("Nhãn (Đáp án thực tế) cảu mẫu đầu tiên:", digits.target[0])
    # Lấy fias trị nhãn chính xác của mẫu thứ 2 ditgits.target[1]
    print ("Nhãn (Đáp án thực tế) cảu mẫu 2:", digits.target[1])
    # khởi tạo mô hình học máy, vd: Logistic cho bài toán phân loại
    model= LogisticRegression(max_iter=10000)

    # huấn luyện mô hình (fit model với dữ liệu)
    model.fit(digits.data, digits.target)

    # Dự đoán với mẫu đầu tiên
    prediction= model.predict([digits.data[0]])
    # digits.data[2]: Lấy tập đặc trưng (các giá trị pixel) của mẫu giá trị thứ 2 để dự đoán
    prediction_2 = model.predict([digits.data[2]])
    # dự đoán tập dữ liệu số 1 có giá trị là gì
    print ("Mô hình dự đoán đây là số mấy:", prediction_2[0])
    # dự đoán tập dữ liệu số 2 có giá trị là gì
    print ("Mô hình dự đoán đây là số mấy:", prediction[0])
# bai3()

def bai4():
    #tải dữ liệu mẫu
    digits= datasets.load_digits()
    print ("Hình dạng dữ liệu (Số lượng mẫu, số đặc trưng):", digits.data.shape)
    print ("Nhãn (Đáp án thực tế) cảu mẫu đầu tiên:", digits.target[0])
    # Lấy fias trị nhãn chính xác của mẫu thứ 2 ditgits.target[1]
    print ("Nhãn (Đáp án thực tế) cảu mẫu 2:", digits.target[1])
    # khởi tạo mô hình học máy, vd: Logistic cho bài toán phân loại
    model= LogisticRegression(max_iter=10000)

    # huấn luyện mô hình (fit model với dữ liệu)
    model.fit(digits.data, digits.target)

    # Dự đoán với mẫu đầu tiên
    prediction= model.predict([digits.data[0]])
    # digits.data[2]: Lấy tập đặc trưng (các giá trị pixel) của mẫu giá trị thứ 2 để dự đoán
    prediction_2 = model.predict([digits.data[2]])
    # dự đoán tập dữ liệu số 1 có giá trị là gì
    print ("Mô hình dự đoán đây là số mấy:", prediction_2[0])
    # dự đoán tập dữ liệu số 2 có giá trị là gì
    print ("Mô hình dự đoán đây là số mấy:", prediction[0])

    # 4. Hien thi mau du lieu dau tien duoi dang hinh anh
    image = digits.images[0]

    plt.figure(figsize=(4, 4))

    plt.imshow(image, cmap="gray")

    plt.title(f"Chu so: {digits.target[0]}")

    plt.axis("off")

    plt.show()
bai4()