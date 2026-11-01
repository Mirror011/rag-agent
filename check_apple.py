# 存成 check_apple.py
import pymupdf, re
doc = pymupdf.open("data/apple-2025.pdf")
text = "".join(p.get_text() for p in doc)
doc.close()
for line in text.split("\n"):
    if re.match(r"^\s*item\s+\d", line, re.I):
        print(repr(line.strip()))