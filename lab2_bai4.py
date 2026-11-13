from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris



def logistic_regression_pipeline():
    iris=load_iris()
    X=iris.data
    y=iris.target

    #Chia dữ liệu theo tỷ lệ 80:20, cố định random_state
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    print("Kich thước tạo X_train: ",X_train.shape)
    print("Kich thước tạo X_test: ",X_test.shape)
    print("Kich thước tạo y_train: ",y_train.shape)
    print("Kich thước tạo y_test: ",y_test.shape)

    #Tạo pipeline gồm StandardScaler và LogisticRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    #Huấn luyện mô hình trên tập huấn luyện
    pipeline.fit(X_train, y_train)

    #Đánh giá mô hình trên tập kiểm tra
    accuracy = pipeline.score(X_test, y_test)
    print("Độ chính xác của mô hình trên tập kiểm tra: {:.2f}%".format(accuracy * 100))
logistic_regression_pipeline()