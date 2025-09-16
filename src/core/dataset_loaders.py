
def load_raw_text_data(file_path: str) -> str:
  with open(file_path, "r", encoding="utf-8") as f: 
    raw_text = f.read()
  return raw_text
