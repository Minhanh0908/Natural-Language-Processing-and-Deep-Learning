# Lab 7: Thực hành chuyên sâu về phân tích cú pháp phụ thuộc (Dependency Parsing)

- **File code:** `./notebook/lab7_dependence_parsing.ipynb`

## Phần 1: Giới thiệu và Cài đặt

Cài đặt và sử dụng `spaCy` - thư viện NLP để khám phá kỹ thuật phân tích cú pháp phụ thuộc.

## Phần 2: Phân tích câu và Trực quan hóa

- **Câu ví dụ**: "The quick brown fox jumps over the lazy dog."
- Từ gốc (ROOT) của câu là: `jumps` (VERB). Động từ chính, biểu thị hành động của câu.
- `jumps` có các từ phụ thuộc (dependent) là:
  - `fox` -> quan hệ **nsubj** (nomial subject - chủ ngữ). `fox` là chủ ngữ của động từ `jumps`.
  - `over` -> quan hệ **prep** (prepositional modifier - giới từ). `over` là giới từ chỉ quan hệ nơi chốn liên kết với động từ `jumps`
- `fox` là head của các từ:
  - `The` -> quan hệ **det** (mạo từ xác định cho danh từ `fox`)
  - `quick` và `brown` -> quan hệ **amod** (tính từ bổ nghĩa, mô tả đặc điểm của `fox`)

## Phần 3: Truy cập các thành phần trong cây phu thuộc

## TEXT | DEP | HEAD TEXT | HEAD POS | CHILDREN

Apple | nsubj | looking | VERB | []
is | aux | looking | VERB | []
looking | ROOT | looking | VERB | ['Apple', 'is', 'at']
at | prep | looking | VERB | ['buying']
buying | pcomp | at | ADP | ['startup']
U.K. | compound | startup | NOUN | []
startup | dobj | buying | VERB | ['U.K.', 'for']
for | prep | startup | NOUN | ['billion']
$ | quantmod | billion | NUM | []
1 | compound | billion | NUM | []
billion | pobj | for | ADP | ['$', '1']

## Phần 4: Duyệt cây phụ thuộc để trích xuất thông tin

### 4.1. Bài toán: Tìm chủ ngữ và tân ngữ của một động từ

- Tìm các cặp (chủ ngữ, động từ, tân ngữ) trong câu "The cat chased the mouse and the dog watched them."
- Kết quả:
  Found Triplet: (cat, chased, mouse)
  Found Triplet: (dog, watched, them)

### 4.2. Bài toán: Tìm các tính từ bổ nghĩa cho một danh từ

- **Câu ví dụ:** "The big, fluffy white cat is sleeping on the warm mat."
- Kết quả:
  Danh từ 'cat' được bổ nghĩa bởi các tính từ: ['big', 'fluffy', 'white']
  Danh từ 'mat' được bổ nghĩa bởi các tính từ: ['warm']

## Phần 5: Bài tập tự luyện:

### Bài 1:

- Hàm `find_main_verb(doc)` nhận vào một đối tượng Doc của spaCy và trả về Token là động từ chính.

```python
def find_main_verb(doc):
  for token in doc:
    if token.dep_ == "ROOT" and token.pos_ == "VERB":
      return token.text
  return None
```

- **Câu ví dụ:** "The dog barked loudly at the stranger."
- **Kết quả:** Động từ chính của câu là: barked

### Bài 2:

- Hàm `extract_noun_chunks(doc)` tìm cụm danh từ thủ công không dùng `doc.noun_chunks`

```python
def extract_noun_chunks(doc):
  noun_chunks = []
  for token in doc:
    if token.pos_ == "NOUN":
      chunk_tokens = [token.text]
      for child in token.children:
        if child.dep_ in ("amod", "det", "compound"):
          chunk_tokens.append(child.text)
      chunk_tokens.sort(key=lambda x: doc.text.index(x))
      noun_chunk = " ".join(chunk_tokens)
      noun_chunks.append(noun_chunk)
  return noun_chunks
```

- **Câu ví dụ:** "The beautiful garden behind the old house has many colorful flowers."
- **Kết quả:** Các cụm danh từ trong câu là:
  The beautiful garden
  the old house
  many colorful flowers

### Bài 3:

- Hàm `get_path_to_root(token)` tìm chuỗi các token trên đường đi từ một từ bất kỳ đến ROOT.

```python
def get_path_to_root(token):
    path = [token]
    while token.head != token:
        token = token.head
        path.append(token)
    return path
```

- **Câu ví dụ:** "The teacher gave the students difficult homework yesterday."
- **Kết quả:** Đường đi từ "difficult" đến root: difficult → homework → gave
