# NLP Labs Repository

Repo này được xây dựng để thực hiện các bài lab của môn NLP&DL, bao gồm:

- **Lab 1:** Tokenizer cơ bản (_SimpleTokenizer, RegexTokenizer_)
- **Lab 2:** CountVectorizer (_Bag-of-Words_) để biểu diễn văn bản dưới dạng vector số
- **Lab 3:** Word Embedding - biểu diễn từ dưới dạng dense-vector (Word2Vec)
- **Lab 4:** Text classification - phân loại văn bản bằng pipeline sử dụng các kỹ thuật tiền xử lý từ các lab trước, sử dụng các model học máy đơn giản.
- **Lab 5:** RNNs và các bài toán: Sử dụng các mô hình RNNs, LSTMs và GRUs để giải quyết các bài toán token classification, POS và NER.
- **Lab 6:** Introduction to Transformers: sử dụng các mô hình pretrained Transformer để thực hiện các tác vụ NLP cơ bản.

---

## Thông tin học phần

- **Tên môn học:** Xử lý ngôn ngữ tự nhiên và học sâu
- **Mã môn học:** MAT3399
- **Giảng viên:** PGS.TS Lê Hồng Phương
- **Họ và tên:** Trần Minh Anh
- **Mã sinh viên:** 22001234
- **Năm học:** Học kỳ 1 - 2025-2026

## Cấu trúc thư mục

```plaintext
project_root/
│
├── src/ # Mã nguồn chính
│ ├── core/ # Các thành phần & interface cốt lõi
│ │ ├── interfaces.py # Abstract base classes: Tokenizer, Vectorizer,...
│ │ └── dataset_loaders.py # Hàm load dữ liệu văn bản
│ │
│ ├── preprocessing/ # Các tác vụ tiền xử lý văn bản: tokenizer, vectorizer, ...
│ │ ├── simple_tokenizer.py
│ │ └── regex_tokenizer.py
│ │
│ └── representations/ # Biểu diễn văn bản
│ |  └── count_vectorizer.py
| |
│ └── models/ # Các mô hình
├── data/ # Thư mục chứa dữ liệu
│
├── notebook/ # Các file notebook
|
├── tests/ # Các script kiểm thử cho từng lab
│ └── labn_test.py # Ví dụ: lab1_test.py, lab2_test.py ...
│
├── logging/ # Log file kết quả test
│ └── labn_test.log # Ví dụ: lab1_test.log, lab2_test.log ...
│
├── report/ # Báo cáo chi tiết cho từng lab
│ └── Report.md
│
├── results/
|  └── word2vec_ewt.model.md #Kết quả train Word2Vec gensim với dữ liệu UDT
|
├── requirements.txt # Các thư viện cần thiết
└── README.md # Tài liệu mô tả repo
```
