"""财报表格结构化：解析 → SQLite → 精确查询"""
import os
import re
import sqlite3
import pymupdf
import config


# ---------- 解析 ----------
def _clean(cells):
    """去掉空列与纯符号列"""
    return [c.strip() for c in cells if c and c.strip() not in ("", "$", "%", "(", ")")]


def _to_number(s):
    """'(6,659)' -> -6659 ; '65,821' -> 65821 ; 非数字返回 None"""
    s = s.strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "").replace("$", "").strip()
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return None
    v = float(s)
    return -v if neg else v

def _filter_by_magnitude(nums, max_ratio=100):
    """剔除与主数值量级悬殊的项（通常是夹在年份列之间的变动百分比）。

    财报表格存在两种排版：
        Tesla   2025 | 2024 | 2023 | 变动额 | 变动%      变动列在末尾
        Apple   2025 | 6 | 391,035 | 2 | 383,285        变动%夹在年份之间
    后者会使年份与数值错位。同一行的年度数值量级相近，
    故以首个数值为基准，剔除小于其 1/100 的项。
    """
    if len(nums) < 2:
        return nums
    base = abs(nums[0])
    if base == 0:
        return nums
    kept = [n for n in nums if abs(n) == 0 or abs(n) >= base / max_ratio]
    return kept if len(kept) >= 2 else nums      # 过滤过度则放弃，保留原值


def _detect_years(page_text):
    """识别页面上的年份序列（财报通常列三年）"""
    years, seen = [], set()
    for y in re.findall(r"\b(20[0-3]\d)\b", page_text):
        if y not in seen:
            seen.add(y)
            years.append(int(y))
    return years[:3]


def build_db(db_path=None):
    """从 PDF 提取表格，解析为 (科目, 年份, 数值) 存入 SQLite"""
    db_path = db_path or config.TABLE_DB_PATH

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE financials (
            company    TEXT,
            item       TEXT,      -- 科目名，如 Total revenues
            item_lower TEXT,      -- 小写，便于匹配
            year       INTEGER,
            value      REAL,
            page       INTEGER,
            table_id   TEXT,      -- 新增：哪一页的第几张表
            row_idx    INTEGER,   -- 新增：表内第几行
            source     TEXT       -- 原始行文本，便于溯源核对
        )
    """)
    conn.execute("CREATE INDEX idx_item ON financials(item_lower)")
    conn.execute("CREATE INDEX idx_year ON financials(year)")

    rows_added = 0
    for cfg in config.DOCUMENTS:
        if not os.path.exists(cfg['path']):
            print(f"  跳过（文件不存在）：{cfg['path']}")
            continue
        print(f"  解析 {cfg['company']} ...")
        doc = pymupdf.open(cfg['path'])
        for pno in range(len(doc)):
            page = doc[pno]
            try:
                tabs = page.find_tables()
            except Exception:
                continue
            if not tabs.tables:
                continue

            years = _detect_years(page.get_text())
            if not years:
                continue

            for ti, tab in enumerate(tabs.tables):
                table_id = f"{cfg['company'][:4]}_p{pno + 1}t{ti + 1}"
                for row_idx, raw in enumerate(tab.extract()):
                    cells = _clean(raw)
                    if len(cells) < 2:
                        continue
                    item = cells[0]
                    if not re.search(r"[A-Za-z]{3,}", item):
                        continue
                    item = " ".join(item.split())  # 顺手修复换行截断
                    nums = [_to_number(c) for c in cells[1:]]
                    nums = [n for n in nums if n is not None]
                    nums = _filter_by_magnitude(nums)  # 剔除夹在年份间的百分比
                    if not nums:
                        continue
                    for i, year in enumerate(years):
                        if i < len(nums):
                            conn.execute(
                                "INSERT INTO financials VALUES (?,?,?,?,?,?,?,?,?)",
                                (cfg['company'], item, item.lower(), year, nums[i],
                                 pno + 1, table_id, row_idx, " | ".join(cells[:6]))
                            )
                            rows_added += 1

        doc.close()
    conn.commit()
    conn.close()
    print(f"结构化数据库已建立：{rows_added} 条记录 -> {db_path}")
    return rows_added


# ---------- 查询 ----------
def _conn():
    return sqlite3.connect(config.TABLE_DB_PATH)


def query_item(keyword, year=None, limit=8):
    """按科目关键词查询，可指定年份"""
    sql = "SELECT item, year, value, page FROM financials WHERE item_lower LIKE ?"
    args = [f"%{keyword.lower()}%"]
    if year:
        sql += " AND year = ?"
        args.append(int(year))
    sql += " ORDER BY year DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        return c.execute(sql, args).fetchall()


def query_trend(keyword, company=None, limit=12):
    """查询某科目的多年数据，可限定公司"""
    sql = """SELECT item, year, value, page, company FROM financials
             WHERE item_lower LIKE ?"""
    args = [f"%{keyword.lower()}%"]
    if company:
        sql += " AND company = ?"
        args.append(company)
    sql += " ORDER BY company, item, year DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        return c.execute(sql, args).fetchall()


def format_results(rows):
    """转成给大模型读的文本，按 (公司,科目,年份) 去重"""
    if not rows:
        return ""
    seen, lines = set(), ["[结构化财务数据]"]
    for r in rows:
        item, year, value, page = r[0], r[1], r[2], r[3]
        comp = r[4] if len(r) > 4 else ""
        key = (comp, item, year)
        if key in seen:
            continue
        seen.add(key)
        prefix = f"{comp} | " if comp else ""
        lines.append(f"{prefix}{item} | {year}年 | {value:,.0f}（百万）| 第{page}页")
    return "\n".join(lines)

def build_hierarchy():
    """基于表内位置构建层级：Total 行之前、上一个 Total 之后的数据行即为其子项"""
    conn = sqlite3.connect(config.TABLE_DB_PATH)
    conn.execute("DROP TABLE IF EXISTS hierarchy")
    conn.execute("""
        CREATE TABLE hierarchy (
            parent_item TEXT, child_item TEXT,
            table_id TEXT, page INTEGER,
            child_row INTEGER
        )
    """)

    tables = [r[0] for r in conn.execute(
        "SELECT DISTINCT table_id FROM financials").fetchall()]

    for tid in tables:
        rows = conn.execute(
            """SELECT DISTINCT row_idx, item, page FROM financials
               WHERE table_id = ? ORDER BY row_idx""", (tid,)).fetchall()

        buffer = []
        last_total = None  # 上一个 Total 行，候选的下级汇总项
        for row_idx, item, page in rows:
            low = item.lower()
            if any(k in low for k in ("margin", "rate", "percent", "%")):
                continue

            if low.startswith("total"):
                # 若上一个 Total 与当前 Total 共享关键词，则视为其子项
                if last_total:
                    prev_words = set(last_total[1].lower().replace("total", "").split())
                    cur_words = set(low.replace("total", "").split())
                    if prev_words & cur_words:  # 有共同词才算父子
                        buffer.append(last_total)

                is_cost = "cost" in low
                for c_row, c_item in buffer:
                    # 成本类汇总项只收成本行，避免同表收入行被吸收
                    if is_cost and "cost" not in c_item.lower():
                        continue
                    conn.execute("INSERT INTO hierarchy VALUES (?,?,?,?,?)",
                                 (item, c_item, tid, page, c_row))
                last_total = (row_idx, item)
                buffer = []
            else:
                buffer.append((row_idx, item))

    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM hierarchy").fetchone()[0]
    conn.close()
    print(f"层级关系已建立：{n} 对父子关系")
    return n


def query_children(parent_keyword, year=None):
    """查询某汇总科目的构成项（限定同表同行，彻底避免同名串台）"""
    with _conn() as c:
        sql = """
            SELECT DISTINCT f.item, f.year, f.value, f.page
            FROM hierarchy h
            JOIN financials f
              ON f.item = h.child_item
             AND f.table_id = h.table_id
             AND f.row_idx = h.child_row
            WHERE h.parent_item LIKE ?
        """
        args = [f"%{parent_keyword}%"]
        if year:
            sql += " AND f.year = ?"
            args.append(int(year))
        sql += " ORDER BY f.year DESC, ABS(f.value) DESC LIMIT 30"
        return c.execute(sql, args).fetchall()

def validate_hierarchy(tolerance=0.01):
    """
    用会计恒等式校验层级：子项之和应等于父项。
    返回 (通过数, 总数, 失败明细)
    """
    conn = sqlite3.connect(config.TABLE_DB_PATH)

    groups = conn.execute("""
        SELECT DISTINCT h.parent_item, h.table_id, f.year
        FROM hierarchy h
        JOIN financials f ON f.item = h.parent_item AND f.table_id = h.table_id
    """).fetchall()

    passed, failed = 0, []
    for parent, tid, year in groups:
        pv = conn.execute("""
            SELECT value FROM financials
            WHERE item = ? AND table_id = ? AND year = ? LIMIT 1
        """, (parent, tid, year)).fetchone()
        if not pv:
            continue
        parent_val = pv[0]

        children = conn.execute("""
            SELECT DISTINCT f.item, f.value FROM hierarchy h
            JOIN financials f
              ON f.item = h.child_item
             AND f.table_id = h.table_id
             AND f.row_idx = h.child_row
            WHERE h.parent_item = ? AND h.table_id = ? AND f.year = ?
        """, (parent, tid, year)).fetchall()
        if len(children) < 2:
            continue

        child_sum = sum(v for _, v in children)
        if abs(parent_val) < 1e-6:
            continue
        diff_pct = abs(child_sum - parent_val) / abs(parent_val)

        if diff_pct <= tolerance:
            passed += 1
        else:
            failed.append({
                "parent": parent, "year": year, "page": tid,
                "parent_val": parent_val, "child_sum": child_sum,
                "diff_pct": diff_pct * 100,
                "children": [(i, v) for i, v in children],
            })

    conn.close()
    total = passed + len(failed)
    rate = passed / total * 100 if total else 0
    print(f"\n=== 层级校验（会计恒等式）===")
    print(f"可校验组合：{total}　通过：{passed}　失败：{len(failed)}　准确率：{rate:.1f}%")
    return passed, total, failed


def show_failures(failed, n=5):
    """打印失败明细，便于定位问题"""
    print(f"\n--- 失败样例（前 {n} 条）---")
    for f in failed[:n]:
        print(f"\n[{f['page']}] {f['parent']}（{f['year']}年）")
        print(f"  父项值：{f['parent_val']:>14,.0f}")
        print(f"  子项和：{f['child_sum']:>14,.0f}   偏差 {f['diff_pct']:.1f}%")
        for item, val in f["children"][:6]:
            print(f"    - {item[:40]:<42} {val:>12,.0f}")
# 核心科目：实际问答中高频出现的汇总项
CORE_ITEMS = [
    "Total revenues",
    "Total automotive revenues",
    "Total cost of revenues",
    "Total automotive cost of revenues",
    "Total gross profit",
    "Total operating expenses",
    "Total current assets",
    "Total current liabilities",
    "Total liabilities",
    "Total stockholders' equity",
]


def validate_core(tolerance=0.01):
    """只校验高频核心科目——实际问答真正会用到的那些"""
    conn = sqlite3.connect(config.TABLE_DB_PATH)
    passed, failed = 0, []

    for core in CORE_ITEMS:
        groups = conn.execute("""
            SELECT DISTINCT h.parent_item, h.table_id, f.year
            FROM hierarchy h
            JOIN financials f ON f.item = h.parent_item AND f.table_id = h.table_id
            WHERE LOWER(h.parent_item) = LOWER(?)
        """, (core,)).fetchall()

        for parent, tid, year in groups:
            pv = conn.execute("""SELECT value FROM financials
                WHERE item = ? AND table_id = ? AND year = ? LIMIT 1""",
                (parent, tid, year)).fetchone()
            if not pv:
                continue
            children = conn.execute("""
                SELECT DISTINCT f.item, f.value FROM hierarchy h
                JOIN financials f ON f.item = h.child_item
                 AND f.table_id = h.table_id AND f.row_idx = h.child_row
                WHERE h.parent_item = ? AND h.table_id = ? AND f.year = ?
            """, (parent, tid, year)).fetchall()
            if len(children) < 2 or abs(pv[0]) < 1e-6:
                continue

            diff = abs(sum(v for _, v in children) - pv[0]) / abs(pv[0])
            if diff <= tolerance:
                passed += 1
            else:
                failed.append((parent, year, tid, pv[0],
                               sum(v for _, v in children), diff * 100))

    conn.close()
    total = passed + len(failed)
    rate = passed / total * 100 if total else 0
    print(f"\n=== 核心科目校验 ===")
    print(f"可校验组合：{total}　通过：{passed}　失败：{len(failed)}　准确率：{rate:.1f}%")
    for f in failed[:3]:
        print(f"  ✗ [{f[2]}] {f[0]}（{f[1]}）父项{f[3]:,.0f} vs 子项和{f[4]:,.0f}　偏差{f[5]:.1f}%")
    return passed, total, failed

if __name__ == "__main__":
    build_db()
    build_hierarchy()
    print("\n--- 测试：Total automotive revenues 的构成项 ---")
    for r in query_children("Total automotive revenues", year=2025):
        print(r)
    passed, total, failed = validate_hierarchy()
    if failed:
        show_failures(failed)
        validate_core()
