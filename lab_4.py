import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer #tập dữ liệu có sẵn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix


#Bài 1
#Tải dữ liệu
data = load_breast_cancer()
X, y = data.data, data.target

#Chia tập dữ liệu Train (80%) và Test (20%)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Chuẩn hóa dữ liệu (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#Khởi tạo và huấn luyện mô hình Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled,y_train)

#Bài 2
#Dự đoán trên tập Test
y_pred = model.predict(X_test_scaled)

#Đánh giá mô hình
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Recall:", recall_score(y_test,y_pred))

#In báo cáo phân loại
print("Classification Report:\n", classification_report(y_test,y_pred))

#Bài 3
#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

#Trực quan hóa bằng seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


tn, fp, fn, tp = cm.ravel()
#Giải thích
# TN (True Negative):
# Thực tế 0, dự đoán 0 -> đúng
print("TN:", tn)

# FP (False Positive):
# Thực tế 0, dự đoán 1 -> sai
print("FP:", fp)

# FN (False Negative):
# Thực tế 1, dự đoán 0 -> sai
print("FN:", fn)

# TP (True Positive):
# Thực tế 1, dự đoán 1 -> đúng
print("TP:", tp)




