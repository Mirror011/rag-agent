import sqlite3
c = sqlite3.connect("financial_data.db")

print("=== Apple: Total net sales ===")
for r in c.execute("""SELECT year, value, page, table_id, source FROM financials
                      WHERE company='Apple' AND item='Total net sales'
                      ORDER BY page, year DESC"""):
    print(f"  {r[0]} | {r[1]:>12,.0f} | 第{r[2]}页 | {r[3]}")
    print(f"       原始行：{r[4][:90]}")

print("\n=== Apple: Total gross margin ===")
for r in c.execute("""SELECT year, value, page, source FROM financials
                      WHERE company='Apple' AND item='Total gross margin'
                      ORDER BY page, year DESC LIMIT 6"""):
    print(f"  {r[0]} | {r[1]:>12,.0f} | 第{r[2]}页")
    print(f"       原始行：{r[3][:90]}")
c.close()