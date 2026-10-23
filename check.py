import pymupdf
import re

doc = pymupdf.open("data/tsla-20251231.pdf")
full_text = ""
for page in doc:
    full_text += page.get_text()
doc.close()

# 找出所有像 "Item X" 的行
lines = full_text.split("\n")
for i, line in enumerate(lines):
    if re.match(r"^\s*Item\s+\d+", line, re.IGNORECASE):
        print(f"[{i}] {line.strip()}")