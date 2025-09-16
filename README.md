# NLP Labs Repository

Repo này được xây dựng để thực hiện các bài lab của môn NLP&DL, bao gồm:

- **Lab 1:** Tokenizer cơ bản (_SimpleTokenizer, RegexTokenizer_)
- **Lab 2:** CountVectorizer (_Bag-of-Words_) để biểu diễn văn bản dưới dạng vector số

---

## Thông tin học phần

- **Tên môn học:** Xử lý ngôn ngữ tự nhiên và học sâu
- **Mã môn học:** MAT3399
- **Giảng viên:** PGS.TS Lê Hồng Phương
- **Họ và tên:** Trần Minh Anh
- **Mã sinh viên:** 22001234
- **Năm học:** Học kỳ 1 - 2025-2026

## Cấu trúc thư mục

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
│ ├── representations/ # Biểu diễn văn bản
| | └── count_vectorizer.py ....  
│
├── data/ # Thư mục chứa dữ liệu
|
├── tests/ # Các script kiểm thử cho từng lab
| └── labn_test.py ....  
|
├── logging/ # Log file kết quả test
| └── labn_test.log ....
|
├── report/ # Báo cáo chi tiết của từng tuần
│ └── Report.md  
|  
├── requirements.txt # Các thư viện cần thiết
└── README.md # Tài liệu mô tả repo
