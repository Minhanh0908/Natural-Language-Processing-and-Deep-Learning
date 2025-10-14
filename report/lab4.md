# Lab 4: Word Embeddings

## Mô tả công việc:

### Task 1-3: Khai thác và sử dụng mô hình embedding pre-trained (Glove)

- **File:** `src/representations/word_embedder.py`
- **Mục tiêu:** Xây dựng lớp `WordEmbedder` cho phép:
  - Tải mô hình embedding đã huấn luyện sẵn từ `gensim`.
  - Truy xuất vector của một từ.
  - Tính độ tương đồng giữa hai từ.
  - Tìm các từ gần nghĩa nhất.
  - Biểu diễn một tài liệu (document) bằng trung bình cộng các vector từ.
- Các kết quả thử nghiệm nằm trong `logging/lab4_test.log`

- **Cách triển khai:**

1. **Khởi tạo mô hình:** hàm `__init__()`

- Tải mô hình tiền huấn luyện từ gensim.downloader.load("glove-wiki-gigaword-50").
- Xử lý ngoại lệ nếu model không tồn tại hoặc không tải được. In ra danh sách tất cả các mô hình có sẵn trong gensim.

2. **Lấy vector biểu diễn của một từ:** hàm `get_vector()`

- Trả về `self.model[word]`. Từ không có trong từ điển thì trả về None.

3. **Tính độ tương đồng giữa hai từ:** hàm `get_similarity()`

- Lấy 2 vector biểu diễn hai từ từ hàm `get_vector()`.
- Tính độ tương đồng theo công thức cosine similarity.
- Nếu 1 trong 2 từ không lấy được vector thì trả về None.

4. **Tìm các từ gần nghĩa nhất:** hàm `get_most_similar()`

- Trả về top_n từ gần nhất với từ cần tìm.
- Nếu từ đó không tồn tại trong từ điển thì trả về None.

5. **Biểu diễn tài liệu:**

- Tokenize văn bản bằng `RegexTokenizer` trong lab 1.
- Với mỗi từ lấy ra vector (từ nào không có trong từ điển thì thay bằng vector 0 với kích thước bằng số chiều các vector từ). Lấy trung bình tất cả các vector thu được document embedding.

- **Nhận xét:**
  - “king – queen” có độ tương đồng cao, thể hiện mối quan hệ giới tính cùng ngữ cảnh “vua – hoàng hậu”.
  - “king – man” thấp hơn vì “man” là khái niệm rộng hơn.
  - Các từ gần với “computer” trong mô hình pre-trained thường là các từ trong cùng miền ngữ nghĩa (ví dụ: “software”, “desktop”, “system”).
  - Mô hình pre-trained GloVe biểu diễn ngữ nghĩa khá tự nhiên và chính xác do đã được huấn luyện trên kho dữ liệu lớn

### Bonus Task: Huấn luyện mô hình Word2Vec từ tập English Universal Dependency Dataset

- **File:** `test/lab4_embedding_training_demo.py`
- **Mục tiêu:** Huấn luyện mô hình Word2Vec trên dữ liệu `en_ewt-ud-train.txt`
- **Cách triển khai:**

1. Đọc file dữ liệu raw text. Tiền xử lý văn bản bằng `gensim.utils.simple_preprocess` để chuyển văn bản thành chữ thường, loại bỏ các ký tự đặc biệt, tách văn bản thành các token
2. Dùng `gensim.models.Word2Vec` để huấn luyện mô hình. Với các tham số

- `sentences`: Dữ liệu huấn luyện, dạng list các câu gồm các từ.
- `vector_size=100`: Kích thước vector nhúng cho mỗi từ.
- `window=5`: Số từ ngữ cảnh xung quanh một từ cần xét.
- `min_count=2`: Bỏ qua các từ xuất hiện ít hơn 2 lần trong tập huấn luyện.

3. Lưu mô hình vào `results/word2vec_ewt.model`
4. Kiểm thử mô hình bằng:

- Tìm các từ tương tự "computer".
- Làm phép analogy: "man->king::woman->?".
- **Nhận xét:**
  - Các từ tương tự “computer” không hoàn toàn liên quan ngữ nghĩa (ví dụ “visa”, “spain”), chứng tỏ mô hình tự huấn luyện chưa đủ dữ liệu để học mối quan hệ ngữ nghĩa chuẩn.
  - Analogy test: kết quả “chris” (tên nam giới) → sai ngữ nghĩa, do dữ liệu không đủ phong phú và mô hình chưa học được quan hệ “giới tính” giữa từ.
    → So với pre-trained GloVe, mô hình Word2Vec tự huấn luyện có chất lượng thấp hơn rõ rệt do: tập dữ liệu nhỏ, thiếu ngữ cảnh đa dạng...

### Advanced Task: Scaling Word2Vec with Apache Spark

- **File:** `test/lab4_spark_word2vec_demo.py`
- **Mục tiêu:** Huấn luyện mô hình Word2Vec phân tán trên tập dữ liệu lớn `c4-train.00000-of-01024-30K.json` bằng PySpark.
- **Các triển khai:**

1. Khởi tạo SparkSession.
2. Đọc và tiền xử lý dữ liệu:

- Lấy cột "text" trong df.
- Chuyển thành chữ thường.
- Xóa ký tự đặc biệt.
- Chia câu thành danh sách các từ.

3.  Huấn luyện mô hình Word2Vec phân tán:
    word2Vec = Word2Vec(
    vectorSize=100,  
     minCount=5,  
     inputCol="words",
    outputCol="result"
    )

        model = word2Vec.fit(df_clean)

4.  Kiểm thử mô hình:

- Từ gần nhất với "computer".
  **Nhận xét:**
- Mô hình Spark Word2Vec học được các quan hệ ngữ nghĩa hợp lý giữa các từ.
- “computer” gần “desktop”, “software” ⇒ phản ánh đúng ngữ cảnh.
- Spark xử lý dữ liệu lớn hiệu quả hơn, cho phép huấn luyện mô hình embedding ở quy mô dữ liệu lớn.
