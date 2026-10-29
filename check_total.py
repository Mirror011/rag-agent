import sqlite3
conn = sqlite3.connect("financial_data.db")
rows = conn.execute("""
    SELECT DISTINCT item FROM financials 
    WHERE item_lower LIKE 'total%' ORDER BY item
""").fetchall()
for r in rows:
    print(r[0])
print(f"\n共 {len(rows)} 个 Total 科目")
conn.close()