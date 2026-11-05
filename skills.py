"""Skill 机制：可复用的「分析方法包」，按需加载（渐进式披露）

目录结构：
    skills/<skill-name>/SKILL.md

SKILL.md 格式（与 Anthropic Agent Skills 规范一致）：
    ---
    name: skill 名称
    description: 一句话说明用途与触发场景
    ---
    正文：完整的方法说明

两层加载：
    第一层  所有 Skill 的 name + description 常驻系统提示词（每个几十字）
    第二层  模型判断需要时，调用 load_skill 工具读取正文（上千字）
这样既不让系统提示词膨胀，又能在需要时给模型足够深的方法指导。
"""
import os

SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")

_CACHE = None


def _parse(path):
    """解析 SKILL.md：拆出头部元信息与正文"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.startswith("---"):
        return None

    # 只按前两个 '---' 切分；正文表格里的 '|---|' 不受影响
    _, front, body = text.split("---", 2)

    meta = {}
    for line in front.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    if "name" not in meta:
        return None

    return {
        "name": meta["name"],
        "description": meta.get("description", ""),
        "body": body.strip(),
        "path": path,
    }


def discover():
    """扫描 skills 目录，返回全部 Skill（结果缓存）"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    found = []
    if os.path.isdir(SKILLS_DIR):
        for folder in sorted(os.listdir(SKILLS_DIR)):
            path = os.path.join(SKILLS_DIR, folder, "SKILL.md")
            if os.path.isfile(path):
                skill = _parse(path)
                if skill:
                    found.append(skill)
    _CACHE = found
    return found


def names():
    return [s["name"] for s in discover()]


def catalog_text():
    """第一层：生成注入系统提示词的 Skill 目录（只有名字与描述）"""
    items = discover()
    if not items:
        return ""
    lines = ["[可用 Skill]（分析方法包。问题属于其适用范围时，"
             "先调用 load_skill 加载完整方法，再按方法执行）"]
    for s in items:
        lines.append(f"- {s['name']}：{s['description']}")
    return "\n".join(lines)


def load(name):
    """第二层：返回某个 Skill 的完整正文；不存在则返回 None"""
    for s in discover():
        if s["name"] == name:
            return s["body"]
    return None
