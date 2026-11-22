from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = load_digits()
X = data.data
y = data.target

#Chia dữ liệu theo tỷ lệ 80:20, cố định random_state
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train=scaler.fit_transform(X_train) #chuẩn hóa tập huấn luyện
X_test=scaler.transform(X_test) #chuẩn hóa tập kiểm tra

model = LogisticRegression(max_iter=10000).fit(X_train,y_train) #huấn luyện mô hình hồi quy

y_pred = model.predict(X_test) #dự đoán dữ liệu kiểm tra

#Đánh giá mô hình
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Classification Report:", classification_report(y_test,y_pred))

#Tìm các dự đoán sai
wrong_indices = []

for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        wrong_indices.append(i)

#In ra 5 mẫu sai
num_samples = min(5, len(wrong_indices))

for i in range(num_samples):
    index = wrong_indices[i]

    print(f"\nSample {i + 1}")
    print("Index       :", index)
    print("Actual label:", y_test[index])
    print("Predicted  :", y_pred[index])