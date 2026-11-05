---
name: profitability-analysis
description: 分析单家公司的盈利能力，包括毛利率、营业利润率、净利率及其变化原因。当用户问"赚钱能力""利润率""毛利率变化""盈利质量"时使用。
---

# 盈利能力分析方法

## 一、取数（尽量一次取齐）

用 query_financials 查询以下科目。关键词支持模糊匹配，短词更稳：

| 指标 | 推荐关键词 | Tesla 常见科目名 | Apple 常见科目名 |
|---|---|---|---|
| 营业收入 | revenues / net sales | Total revenues | Total net sales |
| 毛利 | gross | Total gross profit | Gross margin（注意：是金额） |
| 营业利润 | operations / operating | Income from operations | Operating income |
| 净利润 | net income | Net income attributable to common stockholders | Net income |

某科目查不到时，换更短的关键词重试一次；仍查不到就在答案中说明缺失，不得估算。

## 二、计算（一次 calculate 完成）

对每一年分别计算，所有表达式用分号放进同一次 calculate：

- 毛利率 = 毛利 / 营收 × 100
- 营业利润率 = 营业利润 / 营收 × 100
- 净利率 = 净利润 / 营收 × 100

## 三、拆解：利润率变化从哪里来

1. **毛利率变化** → 看价格与成本：营收下降而成本降得更少，说明价格承压
2. **毛利率与营业利润率的差距** = 期间费用率（研发、销售及管理费用）；差距扩大说明费用在侵蚀利润
3. **营业利润率与净利率的差距** → 利息、投资收益、所得税或一次性项目
4. **分部拆解**（用 get_breakdown 或 query_financials 查分部数据）：
   - Tesla：分别计算汽车业务与能源业务的毛利率（关键词 "gross profit"）
   - Apple：分别计算产品（Products）与服务（Services）的毛利率；服务毛利率通常显著高于产品

## 四、常见陷阱

- Apple 利润表中的 "Gross margin" 是毛利**金额**而非毛利率；同名但数值小于 100 的行可能是百分比，计算前先核对量级
- Tesla 的监管积分（regulatory credits）几乎没有对应成本，会抬高汽车业务毛利率。需要反映真实造车盈利时，计算：
  剔除积分后的汽车毛利率 = (汽车毛利 − 监管积分收入) / (汽车收入 − 监管积分收入) × 100
- 不要只看单一利润率：毛利率上升而营业利润率下降，说明费用增长快于毛利增长

## 五、输出格式

1. 一张表：年份 × 三个利润率
2. 不超过三条结论，每条都要说明变化原因（来自第三步）
3. 末尾标注数据来源（公司、页码）
