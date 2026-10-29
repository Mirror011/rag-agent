import pymupdf

doc = pymupdf.open("data/tsla-20251231.pdf")

total = 0
for pno in range(len(doc)):
    page = doc[pno]
    tabs = page.find_tables()
    if tabs.tables:
        total += len(tabs.tables)
        # 只详细看前几页有表格的情况
        if total <= 3:
            for ti, tab in enumerate(tabs.tables):
                data = tab.extract()
                print(f"\n=== 第 {pno+1} 页 表{ti+1} ({len(data)}行 x {len(data[0]) if data else 0}列) ===")
                for row in data[:6]:
                    print([c[:22] if c else "" for c in row])

print(f"\n全文档共检测到 {total} 张表格")
doc.close()