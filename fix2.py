import io, re
p = 'table_db.py'
s = io.open(p, encoding='utf-8').read()

# 1. build_db 签名
s = s.replace("def build_db(pdf_path=None, db_path=None):",
              "def build_db(db_path=None):")
s = s.replace("    pdf_path = pdf_path or config.PDF_PATH\n", "")

# 2. 打开 PDF 改为遍历多公司（加一层循环）
s = s.replace(
    "    doc = pymupdf.open(pdf_path)\n    rows_added = 0\n\n    for pno in range(len(doc)):",
    "    rows_added = 0\n    for cfg in config.DOCUMENTS:\n"
    "        if not os.path.exists(cfg['path']):\n"
    "            print(f\"  跳过（文件不存在）：{cfg['path']}\")\n"
    "            continue\n"
    "        print(f\"  解析 {cfg['company']} ...\")\n"
    "        doc = pymupdf.open(cfg['path'])\n"
    "        for pno in range(len(doc)):")

# 3. 循环体整体缩进 +4（从 page = doc[pno] 到 doc.close() 之前）
lines = s.split('\n')
out, inside = [], False
for ln in lines:
    if ln.strip().startswith('page = doc[pno]'):
        inside = True
    if ln.strip() == 'doc.close()' and inside:
        out.append('        doc.close()')
        inside = False
        continue
    out.append(('    ' + ln) if (inside and ln.strip()) else ln)
s = '\n'.join(out)

# 4. 公司名与 table_id 带公司前缀
s = s.replace("(config.DOC_COMPANY, item, item.lower()", "(cfg['company'], item, item.lower()")
s = s.replace('table_id = f"p{pno+1}t{ti+1}"', 'table_id = f"{cfg[\'company\'][:4]}_p{pno+1}t{ti+1}"')

io.open(p, 'w', encoding='utf-8').write(s)
print("改写完成，检查：")
print("  多公司循环:", "for cfg in config.DOCUMENTS" in s)
print("  公司名取值:", "cfg['company'], item" in s)
print("  table_id  :", "cfg['company'][:4]" in s)