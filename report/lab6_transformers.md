# Lab 6: Giới thiệu về Transformer

- Mục tiêu:
  - Ôn lại kiến thức cơ bản về kiến trúc Transformer.
  - Sử dụng các mô hình Transformer tiền huấn luyện (pretrained models) để thực hiện các tác vụ NLP cơ bản.
  - Làm quen với thư viện transformers của Hugging Face.
- File code: `notebook/lab6_intro_transformer.ipynb`

## Bài 1: Masked Language Modeling:

- Tác vụ: che một vài từ trong câu bằng token đặc biệt ([MASK]) và yêu cầu mô hình dự đoán xem từ đó là gì.
- Mô hình sử dụng: `bert-base-uncased` qua pipeline `fill-mask` của hugging-face.
- **Câu hỏi:**

1. Mô hình đã dự đoán đúng từ capital không?
2. Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ này?

- **Trả lời:**

1. Mô hình dự đoán từ [MASK] là `capital` với xác suất cao nhất 99%
2. Mô hình Encoder-only phù hợp vì:

- Cách huấn luyện: Masked Language Modeling (MLM): Một số từ trong câu bị che bằng [MASK]. Mô hình nhìn toàn bộ ngữ cảnh (cả trước và sau từ bị mask) để dự đoán từ đó.

- Đặc điểm: Nhìn hai chiều (bidirectional) → có thể hiểu rõ ngữ cảnh xung quanh từ bị che. Thích hợp cho hiểu ngôn ngữ (language understanding) hơn là sinh văn bản.

## Bài 2: Next Token Prediction:

- Tác vụ: yêu cầu mô hình sinh ra phần tiếp theo của một đoạn văn bản cho trước.
- Mô hình: `gpt-2` qua pipeline `text-generation` của hugging-face.
- **Câu hỏi:**

1. Kết quả sinh ra có hợp lý không?
2. Tại sao các mô hình Decoder-only như GPT lại phù hợp cho tác vụ này?

- **Trả lời:**

1. Kết quả sinh ra từ GPT-2:
   "The best thing about learning NLP is that I can always be sure that I'm going to be the right person to get it done. And that I've got the right people in my life who can help me.

I've always been a perfectionist. I've always had a love for creativity. I've always thought I could create things that I could make my parents proud of, but I've always been kind of an impatient person and a big proponent of the status quo. So, you know, I think the best thing I can do for myself is to be a perfectionist and to be not a little bit impatient with myself.

It's like, I've always been an impatient person. I've always been a big proponent of the status quo. I've always believed that you can create things that you can make your parents proud of. So, you know, I'm always in love with getting things done.

And I've always been a big proponent of the status quo. I've always believed that you can produce things that you can make your parents proud of. So, you know, I've always been in love with getting things done. I think the best thing I can do for myself is to try to be a perfectionist.

Brett McKay:"

- Về ngữ pháp và ý nghĩa, văn bản được sinh ra về cơ bản hợp lý: GPT-2 nối câu mồi thành một chuỗi dài với các câu đúng cú pháp.

- Tuy nhiên, GPT-2 đôi khi sinh ra văn bản dài, lặp lại, hoặc lan man. Ví dụ ở đây xuất hiện nhiều câu lặp “I’ve always been…”, “So, you know…”, điều này là đặc trưng của GPT-2 nhỏ (không có khả năng kiểm soát dài hạn).

2. Mô hình Decoder-only phù hợp vì:

- Cách huấn luyện: Next Token Prediction (dự đoán từ tiếp theo). Mô hình được huấn luyện chính xác để dự đoán từ tiếp theo dựa trên ngữ cảnh trước đó, nên khi đưa câu mồi, nó sinh ra tiếp nối một cách tự nhiên.
- Đặc điểm: Unidirectional (một chiều) → chỉ biết các từ trước đó. Kiến trúc unidirectional đảm bảo sinh văn bản theo thứ tự, giống cách con người viết và đọc.

## Bài 3: Sentence Representation:

- Tác vụ: chuyển đổi một câu thành một vector số có chiều dài cố định, nắm bắt được ngữ nghĩa của câu đó. Lấy trung bình cộng của các vector đầu ra của tất cả các token trong câu.
- Mô hình: `bert-base-uncased`
- **Câu hỏi:**

1. Kích thước (chiều) của vector biểu diễn là bao nhiêu? Con số này tương ứng với
   tham số nào của mô hình BERT?
2. Tại sao chúng ta cần sử dụng attention_mask khi thực hiện Mean Pooling?

- **Trả lời:**

1. Kích thước của vector: `torch.Size([1, 768])`

- Nghĩa là 1 câu được biểu diễn bằng 1 vector 768 chiều.
- Con số 768 tương ứng với hidden size của mô hình BERT-base (bert-base-uncased). Đây là số chiều của mỗi token embedding trong layer cuối cùng của BERT.

2. Cần phải thực hiện `attention_mask` khi thực hiện Mean pooling vì:

- Khi tokenize, các câu thường được padding để cùng độ dài. Các token padding không chứa thông tin ngữ nghĩa, nếu tính trung bình cùng với token thật sẽ làm vector biểu diễn sai lệch.
- `attention_mask` có giá trị 1 với token thật, 0 với token padding. Khi Mean Pooling, ta nhân token embedding với mask và chia cho tổng mask.
- Kết quả là trung bình chỉ trên các token thực, bỏ qua token padding, đảm bảo vector biểu diễn chính xác ngữ nghĩa câu.
