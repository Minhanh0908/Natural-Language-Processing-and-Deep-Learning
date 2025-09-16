# 1. Mô tả công việc:

## Lab1: Tokenizer

- Cài đặt hai lớp:
  - **Simple Tokenizer**: tách từ dựa trên dấu
    - Các bước:
      - Lặp qua _string.punctuation_ để chèn dấu cách quanh các ký tự đặc biệt.
      - Dùng _split()_ và _lower()_ để chuyển văn bản về chữ thường và phân tách thành các token.
  - **Regex Tokenizer:** sử dụng biểu thức chính quy để tách token linh hoạt hơn.
    - Regex mẫu: _\w+|[^\w\s]_ (tách từ, số và giữ lại dấu câu riêng biệt)
    - Trả về danh sách token thông qua _re.findall_

## Lab2: CountVectorizer

- Cài đặt lớp **CountVectorizer** sử dụng mô hình **Bag-of-Words**
  - Các phương thức:
    - **fit():** xây dựng vocabulary từ corpus
    - **transform():** chuyển đổi văn bản thành vector số đếm token (document-term-matrix)
    - **fit_transform():** thực hiện fit + transform cùng lúc.
  - Các bước xử lý:
    - Tokenize corpus bằng RegexTokenizer ở lab1.
    - Dùng _set()_ để lấy các token duy nhất.
    - Gán chỉ số cho từng token để tạo từ điển _vocabulary\__
    - Với mỗi documnet -> tạo vector có độ dài bằng số lượng từ vựng -> tăng dần số đếm khi gặp token.

# 2. Kết quả chạy

Các kết quả chạy nằm trong thư mục logging

- lab1_test.log
- lab2_test.log

# 3. Giải thích kết quả

- So sánh Tokenizer:

  - SimpleTokenizer phù hợp cho văn bản ngắn, đơn giản.

  - RegexTokenizer chính xác hơn SimpleTokenizer vì có thể tách từ dựa theo pattern, xử lý tốt số, ký hiệu, dấu câu.

- CountVectorizer:

  - Kết quả vector thể hiện số lần xuất hiện của mỗi token trong từng document.
  - Trên corpus nhỏ, dễ quan sát vector.
  - Trên UD_English-EWT, vocabulary rất lớn → document-term matrix thưa -> tốn nhiều tài nguyên -> không hiệu quả.

- Khó khăn:

  - Văn bản chứa ký hiệu lạ hoặc tiếng nước ngoài.

- Document-term matrix quá lớn khi áp dụng trên dataset thực tế.

# 4. Ghi chú:

Trong quá trình thực hiện lab có sự trợ giúp của AI.
