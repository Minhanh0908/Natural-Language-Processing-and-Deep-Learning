# 1. Lab 5: Làm quen với PyTorch

- Hiểu cách PyTorch tự động tính toán đạo hàm (gradient) thông
  qua autograd.
- Biết cách xây dựng một mạng nơ-ron đơn giản bằng cách kế thừa lớp
  torch.nn.Module.
- Làm quen với hai lớp (layer) cơ bản: nn.Linear và nn.Embedding.
- **File code**: `test/lab5_pytorch_introduction.ipynb`

# 2. Lab 5: Phân loại Văn bản Với Mạng Nơ-ron Hồi Quy (RNN/LSTM)

## 2.1 Tổng quan:

- Xây dựng, huấn luyện và so sánh hiệu năng giữa các mô hình phân loại văn bản trên bộ dữ liệu `hwu` (cột `text`: câu lệnh truy vấn người dùng, cột `category`: nhãn ý định)
  1. TF-IDF + LogisticRegression
  2. Word2Vec (vecto trung bình) + Dense Layer
  3. Embedding Layer (pre-trained) + LSTM
  4. Embedding Layer (from scratch) + LSTM
- **File code**: `test/lab5_rnns_text_classification.ipynb`

## 2.2 Các bước thực hiện:

**Bước 1: Tiền xử lý dữ liệu:**

- Đọc dữ liệu train, val, test.
- Mã hóa nhãn (côt `category`) thành dạng số.

**Bước 2: Xây dựng các mô hình phân loại văn bản:**

1. **Pipeline TF-IDF + Logistic Regression:**

- **Feature extraction:** Sử dụng TfidfVectorizer(max_features=5000) để biểu diễn câu dưới dạng vector tần suất tf-idf.
- **Classifier:** LogisticRegression(max_iter=1000) huấn luyện trên đặc trưng tf-idf.

2. **Pipeline Word2Vec(Average) + Dense Layer:**

- **Huấn luyện Word2Vec:**  
  Sử dụng `Word2Vec` với các tham số:

  - `vector_size=100`: Kích thước vector biểu diễn từ.
  - `window=5`: Kích thước cửa sổ ngữ cảnh.
  - `min_count=2`: Bỏ qua các từ xuất hiện ít hơn 2 lần.
  - `epochs=50`: Số vòng lặp huấn luyện trên toàn tập dữ liệu.

- **Trích xuất đặc trưng câu:**  
  Với mỗi câu, lấy trung bình (mean) của tất cả vector từ trong câu để được vector biểu diễn câu.

- **Kiến trúc mô hình:** Mô hình Sequential với các tầng
  - `Dense(128, activation='relu')`: Học biểu diễn phi tuyến.
  - `Dropout(0.2)`: Ngẫu nhiên loại bỏ 20% neuron trong mỗi bước tính để giảm overfitting.
  - `Dense(num_classes, softmax)`: Dự đoán phân phối xác suất trên các lớp.
  - Optimizer: Adam
  - Loss: sparse_categorical_crossentropy

3. **Embedding Pre-trained +LSTM:**

- **Tiền xử lý chuỗi**:

  - Sử dụng `Tokenizer(num_words=5000, oov_token='<UNK>')` để mã hóa từ thành chỉ số.
  - Dùng `pad_sequences(maxlen=20, padding='post')` để đảm bảo mọi câu có cùng độ dài.

- **Tạo ma trận embedding**:

  - Khởi tạo `embedding_matrix` từ vector Word2Vec đã huấn luyện.
  - `trainable=False` để giữ nguyên trọng số embedding (pre-trained).

- **Kiến trúc mô hình**:
  - **Kiến trúc mô hình**:
  - Embedding(input_dim=5000, output_dim=100, weights=[embedding_matrix], trainable=False): tầng embedding pretrained bằng Word2Vec
  - LSTM(128, dropout=0.5, recurrent_dropout=0.2): tầng
  - Dense(num_classes, activation='softmax')
  - Optimizer: Adam
  - Loss: sparse_categorical_crossentropy
  - EarlyStopping được dùng để tránh overfitting (patience=3, monitor='val_loss').

4. **Embedding from scratch + LSTM:**

- Cấu trúc tương tự mô hình (3), nhưng Embedding layer được học từ đầu (trainable=True).

**Bước 3: Đánh giá, So sánh và Phân tích:**

### **Bảng tổng hợp kết quả đánh giá**

| Pipeline                       | F1-score (Macro) | Test Loss | Inference Time (s/sample) |
| ------------------------------ | ---------------- | --------- | ------------------------- |
| TF-IDF + Logistic Regression   | 0.835298         | —         | 0.000011                  |
| Word2Vec (Avg) + Dense         | 0.793532         | 0.714392  | 0.000101                  |
| Embedding (Pre-trained) + LSTM | 0.829596         | 0.622635  | 0.000272                  |
| Embedding (Scratch) + LSTM     | 0.830693         | 0.842276  | 0.000330                  |

- **Nhận xét**:

  - F1-score của TF-IDF và LSTM tương đương (~0.83), nhưng LSTM có khả năng hiểu ngữ cảnh và thứ tự từ.

  - Mô hình Word2Vec + Dense có F1 thấp hơn (0.79) vì mất thông tin thứ tự.

  - Thời gian suy luận (inference) tăng theo độ phức tạp mô hình: Logistic Regression < Dense < LSTM.

- **Thử với một vài câu "khó"**:
  - **Câu ví dụ**:
    texts = ["i do not want to use the socket anymore",
    "is it going to be sunny or rainy tomorrow",
    "tell me the preparation method for chicken sixty five"]
    intent = le.transform(["iot_wemo_off", "weather_query", "cooking_recipe"])
  - **Kết quả dự đoán**:

| Câu                                                     | Nhãn thật      | TF-IDF + LR    | Word2Vec + Dense  | LSTM (Pre-trained) | LSTM (Scratch) |
| ------------------------------------------------------- | -------------- | -------------- | ----------------- | ------------------ | -------------- |
| _i do not want to use the socket anymore_               | iot_wemo_off   | cooking_recipe | audio_volume_mute | **iot_wemo_off**   | iot_wemo_on    |
| _is it going to be sunny or rainy tomorrow_             | weather_query  | weather_query  | weather_query     | **weather_query**  | weather_query  |
| _tell me the preparation method for chicken sixty five_ | cooking_recipe | cooking_recipe | qa_maths          | **cooking_recipe** | general_quirky |

- **Nhận xét**:

  - Câu 1: chứa từ phủ định “not”, TF-IDF và Word2Vec không hiểu được mối quan hệ phủ định → dự đoán sai. LSTM hiểu thứ tự từ → biết rằng “not want to use” thể hiện ý định tắt thiết bị → đúng iot_wemo_off.
  - Câu 2: câu hỏi thời tiết đơn giản, cả 4 mô hình đều nhận đúng.
  - Câu 3: câu có ý nghĩa phụ thuộc xa (“preparation method ... for chicken”) → LSTM giữ được ngữ cảnh và hiểu đúng mục đích. Word2Vec mất cấu trúc chuỗi nên nhầm sang chủ đề “qa_maths”.

- **Ưu và nhược điểm của từng mô hình**:
  | Mô hình | Ưu điểm | Nhược điểm |
  | -------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
  | **TF-IDF + Logistic Regression** | Đơn giản, huấn luyện nhanh, F1 cao bất ngờ trên dữ liệu nhỏ. | Không hiểu ngữ cảnh, thứ tự từ, không xử lý phủ định. |
  | **Word2Vec (Avg) + Dense** | Học được ngữ nghĩa tổng thể từ embedding. | Mất thông tin thứ tự, độ chính xác giảm ở câu dài hoặc phủ định. |
  | **LSTM (Pre-trained)** | Giữ được thứ tự từ, hiểu ngữ cảnh, xử lý phủ định tốt. | Huấn luyện lâu hơn, yêu cầu embedding tốt. |
  | **LSTM (Scratch)** | Tự học embedding phù hợp với task, tiềm năng cao. | Dễ underfit với dữ liệu nhỏ, cần nhiều epoch hơn để hội tụ. |

## 2.3 Khó khăn trong quá trình làm

- Các mô hình Dense và LSTM ban đầu có F1-score thấp, cần thử tăng số epoch, điều chỉnh các siêu tham số (learning rate, batch size, số lượng neurons) để cải thiện hiệu suất.
- Mặc dù loss giảm đều trong quá trình huấn luyện, mô hình đôi khi bị dừng sớm do cơ chế **EarlyStopping**. Lý do là khi chạy lại mô hình, callback `early_stop` vẫn lưu lại giá trị loss từ lần huấn luyện trước.

# 3. Lab 5: Xây dựng mô hình RNN cho bài toán Part-of-Speeech Tagging

## 3.1 Tổng quan:

- Xây dựng mô hình RNN cho bài toán POS bằng PyTorch trên bộ dữ liệu UD_English-EWT.
- **File code**: `test/lab5_rnn_for_pos_tagging.ipynb`

## 3.2 Các bước thực hiện

**Bước 1: Tiền xử lý dữ liệu:**

- Thực hiện hàm `load_conllu(file_path)` để đọc file `.conllu` từ `data/UD_English-EWT`
- Trích ra 2 cột:
  - token (word)
  - pos tag
- Mỗi câu là một list các tuple (word, tag).
- Ví dụ 5 từ đầu trong câu đầu của tập train: [('Al', 'NNP'), ('-', 'HYPH'), ('Zaman', 'NNP'), (':', ':'), ('American', 'JJ')]

**Bước 2: Xây dựng từ điển:**

- `word_to_ix`: ánh xạ mỗi từ sang một số nguyên (`UNK` = 0 cho nhưng từ chưa có trong từ điển)
- `tag_to_ix`: ánh xạ mỗi nhãn UPOS sang một số nguyên
- Kết quả:

  - 10 từ đầu trong `word_to_ix`: [('<UNK>', 0), ('Al', 1), ('-', 2), ('Zaman', 3), (':', 4), ('American', 5), ('force', 6), ('kill', 7), ('Shaikh', 8), ('Abdullah', 9)]

  - Nhãn upos: [('NNP', 0), ('HYPH', 1), (':', 2), ('JJ', 3), ('NNS', 4), ('VBD', 5), (',', 6), ('DT', 7), ('NN', 8), ('IN', 9), ('.', 10), ('-LRB-', 11), ('MD', 12), ('VB', 13), ('VBG', 14), ('PRP', 15), ('TO', 16), ('-RRB-', 17), ('VBN', 18), ('RP', 19), ('CD', 20), ('VBZ', 21), ('RB', 22), ('NNPS', 23), ('VBP', 24), ('PRP$', 25), ('CC', 26), ('_', 27), ('WP', 28), ('EX', 29), ('WDT', 30), ('RBR', 31), ('PDT', 32), ('JJR', 33), ('WRB', 34), ('JJS', 35), ('``', 36), ("''", 37), ('POS', 38), ('RBS', 39), ('WP$', 40), ('ADD', 41), ('FW', 42), ('LS', 43), ('UH', 44), ('AFX', 45), ('$', 46), ('NFP', 47), ('SYM', 48), ('GW', 49)]

**Bước 3: Thiết lập Dataset và DataLoader:**

- Xây dựng lớp `POSDataset` kết thừa tự `torch.utils.data.Dataset`
- Hàm `collate_fn` sử dụng `pad_sequence` để xử lý độ dài câu bằng padding đảm bảo các câu trong cùng một batch có độ dài bằng nhau.

**Bước 4: Xây dựng mô hình RNN:**

- Xây dựng một mạng nơ-ron RNN-based đơn giản với các layer:

1. `nn.Embedding`: chuyển index của token sang vector embedding (embedding_dim=100).
2. `nn.RNN`: xử lý chuỗi embeddings với hidden_dim=128.
3. `nn.Linear`: ánh xạ đầu ra RNN sang logits ()

**Bước 5: Huấn luyện mô hình**

- Optimizer: `torch.optim.Adam`
- Loss function: `nn.CrossEntropyLoss(ingnore_index=-1)` bỏ quả các token padding khi tính đạo hàm.
- Vòng lặp huấn luyện:
  - Xóa gradients cũ: `optimizer.zero_grad()`
  - Forward pass: `logits = model(sentences)`
  - Tính loss: `loss = criterion(logits, tags)`
  - Backpropagation: `loss.backward()`
  - Cập nhật trọng số: `optimizer.step()`
  - Sau mỗi epoch in ra loss trung bình
- Kết quả huấn luyện sau 5 epoch:
  Epoch 1/5, Training Loss: 1.3354
  Epoch 2/5, Training Loss: 0.7418
  Epoch 3/5, Training Loss: 0.5783
  Epoch 4/5, Training Loss: 0.4825
  Epoch 5/5, Training Loss: 0.4164

**Bước 6: Đánh giá mô hình:**

- Thực hiện hàm `evaluate()` để tính accuracy:
  - Lấy kết quả raw output logits từ model `logits = model(sentences)`
  - Lấy ra nhãn dự đoán sử dụng `torch.argmax(logits, dim=-1)`
  - Tính accuracy bỏ qua các token là padding.
- Lưu loại mô hình có accuracy tốt nhất trên tập dev: `torch.save(model.state_dict(), "best_model.pt")`
- Kết quả sau 5 epochs:
  Epoch [1/5] - Loss: 0.3662 | Train Acc: 0.8910 | Dev Acc: 0.8167
  Epoch [2/5] - Loss: 0.3255 | Train Acc: 0.9006 | Dev Acc: 0.8213
  Epoch [3/5] - Loss: 0.2912 | Train Acc: 0.9118 | Dev Acc: 0.8217
  Epoch [4/5] - Loss: 0.2630 | Train Acc: 0.9209 | Dev Acc: 0.8232
  Epoch [5/5] - Loss: 0.2382 | Train Acc: 0.9291 | Dev Acc: 0.8258

**Bước 7: Dự đoán trên câu mới:**

- Hàm `predict_sentence()` nhận vào một câu mới dạng chuỗi -> ánh xạ các từ thành các số nguyên tưng ướng trong `word_to_ix` -> tạo tensor input đưa vào mô hình -> dự đoán -> trả về list cặp (từ, nhãn dự đoán)

- Ví dụ dự đoán câu mới:
  - Câu: "The cat sits on the mat"
  - Dự đoán: [('the', 'DT'),
    ('cat', 'NN'),
    ('sits', 'NNS'),
    ('on', 'IN'),
    ('the', 'DT'),
    ('mat', 'NN')]
- Câu: "The cat sit on the mat"
- Dự đoán: "[('the', 'DT'),
  ('cat', 'NN'),
  ('sit', 'VBD'),
  ('on', 'IN'),
  ('the', 'DT'),
  ('mat', 'NN')]"
- Nhận xét: cùng là từ "sit" nhưng câu 1 dự đoán sai do chưa xử lý đưa từ về dạng root form.
