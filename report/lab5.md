# Lab 4: Text classifications

## Task 1: Scikit-learn TextClassifier

**Các bước thực hiện:**

1. Khởi tạo lớp **Vectorizer()**:

- Nhận và lưu đối tượng Vectorizer từ interface
- Khởi tạo `_model=None`

2. Triển khai phương thức **fit()**:

- Chuyển danh sách văn bản thành ma trận số bằng `fit_transform()` của vectorizer.
- Khởi tạo và huấn luyện mô hình Logistic bằng `fit()` của \_model.

3. Triển khai phương phức **predict()**:

- Kiểm tra xem mô hình đã được huấn luyện chưa -> lỗi nếu chưa.
- Dùng `transform()` của vectorizer để mã hóa trên dữ liệu mới.
- Trả về kết quả dự đoán.

4. Triển khai phương thức evaluate

- Tính 4 chỉ số đánh giá mô hình phân loại: acccuracy, precision, recall, f1-score bằng hàm pre-built trong scikit-learn.
- Trả về dictionary kết quả.

## Task 2: Basic Test Case

**Các bước thực hiện**:

1. Chuẩn bị dữ liệu:

- texts: mảng các văn bản.
- labels: nhãn tương ứng.

2. Chia dữ liệu train test:

- Sử dụng `train_test_split()` trong scikit-learn: test_size=0.2.

3. Khởi tạo các đối tượng:

- `RegexTokenizer()` tách văn bàn thành từ bằng regex.
- `CountVectorizer()` tạo vector văn bản bằng count vectorizer.
- `TextClassifier()` logistic model phân loại văn bản.

4. Huấn luyện mô hình trên tập train.

- `model.fit(X_train, y_train)`

5. Dự đoán trên tập test

- `y_pred = model.predict(X_test)`

6. Tính các chỉ số đánh giá

- `metrics = model.evaluate(y_test, y_pred)`

**Chạy code**:

- Khởi động môi trường ảo: `.\env\Scripts\activate`
- Install thư viện cần thiết: `pip install -r requirement.txt`
- Chạy code: `python test/lab5_test.py`

**Nhận xét kết quả**:
Kết quả mô hình:
accuracy: 0.5000
precision: 0.5000
recall: 1.0000
f1: 0.6667

- Tập dữ liệu quá nhỏ (chỉ có 6 câu) -> kết quả chạy mô hình không có ý nghĩa -> thử trên tập dữ liệu lớn hơn.

## Task 3: Running the Spark Example

**Các bước thực hiện:**

1. Khởi tạo Spark Session:
2. Đọc và tiền xử lý dữ liệu:

- Đọc file CSV.
- Chuyển nhãn từ -1/1 -> 0/1
- Lọc bỏ dòng có text hoặc sentiment là null
- Chia dữ liệu: 80% train, 20% test.

3. Xây dựng pipeline tiền xử lý:

- Tokenizer: Tách câu thành danh sách từ.
- StopWordsRemover: Loại bỏ từ dừng ("the", "a", "is",..)
- HashingTF: Chuyển danh sách từ thành vector đếm tần sất
- IDF: Tính trọng số TF-IDF để giảm ảnh hưởng của từ phổ biến.

4. Huấn luyện mô hình:

- Sử dụng `LogisticRegression()`
- Gộp tất cả thành `Pipeline` tự động chạy tuần tự từ tiền xử lý đến huấn luyện mô hình.
- Huấn luyện mô hình trên tập train.

5. Đánh giá mô hình:

- Dự đoán trên tập test.
- Dùng `MulticlassClassificationEvaluator` tính Accuracy và F1-score.

**Chạy code**:

- Khởi động môi trường ảo: `.\env\Scripts\activate`
- Install thư viện cần thiết: `pip install -r requirement.txt`
- Chạy code: `python test/lab5_spark_sentiment_analysis.py`

**Nhận xét kết quả:**
Accuracy: 0.7294860234445446
F1 Score: 0.7266221530017497

## Task 4: Model Improvement Experiment

**Các bước thực hiện:**

1. Khởi tạo Spark
2. Đọc và tiền xử lý (giống task 3)
3. Biểu diễn đặc trưng: theo 2 cách

- TF-IDF: HashingTF + IDF.
- Word2Vec: vectorSize=50, minCount=5.

4. Mô hình phân loại: 3 mô hình

- LogisticRegression
- GBTClassifier
- MLP

5. Pipeline:

- Tạo 6 pipeline (3 mô hình x 2 biểu diễn đặc trưng): tiền xử lý -> trích đặc trưng -> huấn luyện -> dự đoán -> tính chỉ số đánh giá Accuracy và F1.

6. In ra kết quả tổng hợp so sánh giữa cách mô hình

**Chạy code**:

- Khởi động môi trường ảo: `.\env\Scripts\activate`
- Install thư viện cần thiết: `pip install -r requirement.txt`
- Chạy code: `python test/lab5_improvement_test.py`

**Nhận xét kết quả:**
Model Accuracy F1 Score
TF-IDF + Logistic Regression 0.7295 0.7266
TF-IDF + GBT 0.7466 0.7174
TF-IDF + MLP 0.7755 0.7736
Word2Vec + Logistic Regression 0.6844 0.6460
Word2Vec + GBT 0.6709 0.6340
Word2Vec + MLP 0.6673 0.6066

- Mô hình baseline: TF-IDF + Logistic Regression
  - Accuracy: 72.95%
  - F1 Score: 72.66%
- Mô hình cải tiến so với baseline: TF-IDF + MLP
  - Accuracy: 77.55%
  - F1 Score: 77.36%
    -> Mô hình MLP có khả năng học các quan hệ phi tuyến giữa các đặc trưng TF-IDF, giúp mô hình nhận diện cảm xúc hiệu quả hơn.
- Các mô hình Word2Vec (Logistic, GBT, MLP) đều cho kết quả kém hơn TF-IDF
  - Accuracy khoảng 0.66 – 0.68, F1-score thấp hơn rõ rệt.
  - Nguyên nhân:
    - Tập dữ liệu gồm các đoạn tweet ngắn, thông tin ngữ cảnh hạn chế.
    - Dữ liệu huấn luyện Word2Vec chưa đủ lớn, nên vector embedding học được không đủ biểu diễn ngữ nghĩa tốt.
    - Trong trường hợp dữ liệu nhỏ, TF-IDF phù hợp hơn vì không cần huấn luyện biểu diễn từ, chỉ dựa trên tần suất xuất hiện từ.
- So sánh 3 mô hình:
  - MLP thể hiện tốt nhất nhờ khả năng mô hình hóa quan hệ phi tuyến.
  - GBT hoạt động ổn định nhưng kém hiệu quả khi đặc trưng rời rạc như TF-IDF.
  - Logistic Regression đơn giản và nhanh, nhưng độ chính xác hạn chế.

## Nguồn tham khảo: CHATGPT
