"""SLARD-31 topic classification: evidence extraction and dynamic candidate selection.

SLARD-31 is a 31-way statute-topic classification task and follows a different
paradigm from the LCR-CN conflict task:

* extraction returns, for each of the 31 topic categories, the verbatim spans
  that support that topic (fields are category names, not element roles);
* the candidate set is every category with at least one supporting span. When
  extraction returns nothing at all the full label space is used as a fallback,
  so a silent extraction failure cannot trivially "solve" the task by
  narrowing the candidates;
* the prediction is a category *name*, matched exactly against the admissible
  set after stripping decorations such as ``环境保护类`` or surrounding prose.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import parsing
from .backends import BaseBackend

CATEGORIES: List[str] = [
    "环境保护", "民政", "能源", "教育", "人权", "牧业", "工业管理", "水利", "法制工作", "知识产权",
    "农业", "旅游", "土地", "林业", "房地产", "体育", "工商管理", "文物文史", "合同", "商贸物资",
    "营商环境优化", "国家安全", "企业", "广告", "劳动工会", "测绘", "军事", "渔业", "价格",
    "建筑业", "文化",
]

CATEGORY_DESC: Dict[str, str] = {
    "环境保护": "生态环境保护、污染防治、生态修复、排污监管、环保责任等内容",
    "民政": "社会救助、社会福利、基层自治、婚姻殡葬、社会组织管理等内容",
    "能源": "电力、煤炭、石油、天然气、新能源、能源开发利用与监管等内容",
    "教育": "学校教育、职业教育、教师、学生、教育管理等内容",
    "人权": "公民合法权益保护、人格权、平等权、平等保护、救济权等内容",
    "牧业": "畜牧养殖、草原、饲料、动物防疫、畜禽管理等内容",
    "工业管理": "工业生产、制造业、工业园区、工业监管、工业发展等内容",
    "水利": "水资源、水工程、防洪、水土保持、河道湖泊管理等内容",
    "法制工作": "立法、执法、司法、普法、备案审查、法律实施、行政规范性文件等内容",
    "知识产权": "专利、商标、著作权、商业秘密、地理标志、知识产权保护等内容",
    "农业": "种植业、农产品、农业生产、农村发展、农业技术推广等内容",
    "旅游": "旅游资源、景区、游客、旅游经营、旅游服务监管等内容",
    "土地": "土地利用、土地征收、土地出让、耕地保护、宅基地、不动产等内容",
    "林业": "森林、林木、林地、野生植物、森林防火、林业资源保护等内容",
    "房地产": "房地产开发、房屋交易、物业管理、住房保障、房屋征收等内容",
    "体育": "体育活动、体育设施、全民健身、竞技体育、体育产业等内容",
    "工商管理": "市场监督、市场主体登记、行政执法、经营监管、反不正当竞争等内容；不包括企业自身经营义务",
    "文物文史": "文物保护、历史文化遗产、档案、地方志、历史建筑等内容",
    "合同": "合同订立、履行、变更、解除、违约责任、协议义务等内容",
    "商贸物资": "商品流通、贸易、物流、储备、物资供应、批发零售等内容",
    "营商环境优化": "优化营商环境、简政放权、政务服务、市场准入、降低制度性交易成本等内容",
    "国家安全": "国家安全、数据安全、网络安全、保密、反间谍、公共安全风险防范等内容",
    "企业": "企业设立、经营、责任、内部管理、生产经营活动等内容；不包括市场监管执法",
    "广告": "广告发布、广告经营、广告内容、广告监管等内容",
    "劳动工会": "劳动关系、工资、社会保险、职业保护、工会、职工权益等内容",
    "测绘": "测绘活动、地图、地理信息、测量标志、测绘成果管理等内容",
    "军事": "国防、军事设施、兵役、军人权益、军民融合等内容",
    "渔业": "水产养殖、渔业资源、渔港、渔船、禁渔、捕捞管理等内容",
    "价格": "价格管理、收费标准、价格监督、价格违法行为等内容",
    "建筑业": "建筑施工、工程建设、施工安全、建设单位、工程质量等内容",
    "文化": "文化事业、文化产业、出版、广播影视、公共文化服务、文化市场等内容",
}

EXTRACT_KEEP_FIELDS = (
    "id",
    "title",
    "clause_content",
    "regulation",
    "category",
    "content",
    "high_level_laws",
)


def empty_extraction() -> Dict[str, List[str]]:
    return {category: [] for category in CATEGORIES}


def is_non_empty(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def get_extracted_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """Locate the category-keyed evidence map inside a prediction row."""
    fields = item.get("extracted_fields")
    if isinstance(fields, dict):
        return fields

    elements = item.get("extracted_elements")
    if isinstance(elements, dict):
        content_elements = elements.get("content_elements")
        if isinstance(content_elements, dict):
            return content_elements
        if any(key in CATEGORIES for key in elements):
            return elements

    return {}


def choose_candidate_categories(
    extracted_fields: Any,
    mode: str = "dynamic",
) -> Tuple[List[str], str]:
    """Return admissible topics plus a trace of which ones fired."""
    if mode == "static":
        return list(CATEGORIES), "static"

    if not isinstance(extracted_fields, dict):
        return list(CATEGORIES), "all_empty"

    candidates = [
        category for category in CATEGORIES if is_non_empty(extracted_fields.get(category))
    ]
    if not candidates:
        return list(CATEGORIES), "all_empty"

    return candidates, ",".join(candidates)


def normalize_category_output(text: str, candidates: Sequence[str]) -> Optional[str]:
    """Match a free-form response to exactly one admissible category name.

    Responses may carry decorations such as ``环境保护类``, quotes, or a short
    explanation. Longest-match-first ordering prevents ``企业`` from shadowing
    ``企业`` inside ``建筑业`` style compounds, and exact equality is tried
    before substring matching.
    """
    if not text:
        return None

    cleaned = text.strip().strip("\"'“”‘’").strip()
    if cleaned in candidates:
        return cleaned

    ordered = sorted(candidates, key=len, reverse=True)

    for category in ordered:
        if cleaned == f"{category}类":
            return category

    hits = [category for category in ordered if category in cleaned]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # Prefer the match that starts earliest and is longest.
        hits.sort(key=lambda cat: (cleaned.find(cat), -len(cat)))
        return hits[0]

    # Last resort: the model may answer with an index into the candidate list.
    match = re.fullmatch(r"\s*(\d+)\s*", cleaned)
    if match:
        index = int(match.group(1))
        if 0 <= index < len(candidates):
            return candidates[index]

    return None


# --------------------------------------------------------------------------- #
# Extraction stage
# --------------------------------------------------------------------------- #
def build_extract_prompt(bundle: Dict[str, Any], title: str, content: str) -> str:
    definitions = "\n".join(
        f"- {category}：{CATEGORY_DESC[category]}" for category in CATEGORIES
    )
    schema = json.dumps(empty_extraction(), ensure_ascii=False, indent=2)

    return (
        bundle["extraction_prompt"]
        .replace("[[CATEGORY_DEFINITIONS]]", definitions)
        .replace("[[EMPTY_SCHEMA]]", schema)
        .replace("[[TITLE]]", title or "")
        .replace("[[CONTENT]]", content or "")
    )


def topic_extract_row(
    backend: BaseBackend,
    bundle: Dict[str, Any],
    item: Dict[str, Any],
    index: int,
) -> Optional[Dict[str, Any]]:
    prompt = build_extract_prompt(
        bundle,
        title=item.get("title", ""),
        content=item.get("clause_content") or item.get("content", ""),
    )

    try:
        raw = backend.generate(bundle["extraction_system_prompt"], prompt)
    except Exception:  # noqa: BLE001
        return None

    payload = parsing.extract_json_object(raw)
    if payload is None:
        return None

    extracted: Dict[str, Any] = {}
    for category in CATEGORIES:
        value = payload.get(category, [])
        if value is None:
            value = []
        elif isinstance(value, str):
            value = [value]
        elif not isinstance(value, list):
            value = [str(value)]
        extracted[category] = value

    row = {field: item.get(field) for field in EXTRACT_KEEP_FIELDS}
    row["_index"] = index
    row["extracted_fields"] = extracted
    return row


def topic_extract_placeholder(item: Dict[str, Any], index: int, reason: str) -> Dict[str, Any]:
    row = {field: item.get(field) for field in EXTRACT_KEEP_FIELDS}
    row["_index"] = index
    payload: Dict[str, Any] = empty_extraction()
    payload["error"] = reason
    row["extracted_fields"] = payload
    return row


def topic_extract_succeeded(row: Dict[str, Any]) -> bool:
    fields = row.get("extracted_fields")
    if not isinstance(fields, dict) or "error" in fields:
        return False
    return all(isinstance(fields.get(category), list) for category in CATEGORIES)


# --------------------------------------------------------------------------- #
# Classification stage
# --------------------------------------------------------------------------- #
def build_classify_prompt(
    bundle: Dict[str, Any],
    item: Dict[str, Any],
    candidates: Sequence[str],
    extracted_fields: Dict[str, Any],
    task_type: str,
) -> str:
    candidate_text = "\n".join(f"- {category}" for category in candidates)

    return (
        bundle["classification_prompt"]
        .replace("[[CANDIDATE_CATEGORIES]]", candidate_text)
        .replace("[[TITLE]]", item.get("title", "") or "")
        .replace("[[CONTENT]]", item.get("clause_content") or item.get("content", "") or "")
        .replace("[[EXTRACTED_FIELDS]]", json.dumps(extracted_fields, ensure_ascii=False, indent=2))
        .replace("[[TASK_TYPE]]", task_type)
    )


def _gold_category(item: Dict[str, Any]) -> Optional[str]:
    gold = item.get("category") or item.get("gold_category")
    return gold if isinstance(gold, str) else None


def topic_classify_row(
    backend: BaseBackend,
    bundle: Dict[str, Any],
    item: Dict[str, Any],
    index: int,
    mode: str = "dynamic",
) -> Optional[Dict[str, Any]]:
    extracted_fields = get_extracted_fields(item)
    candidates, trace = choose_candidate_categories(extracted_fields, mode=mode)

    prompt = build_classify_prompt(bundle, item, candidates, extracted_fields, trace)

    try:
        raw = backend.generate(bundle["classification_system_prompt"], prompt)
    except Exception:  # noqa: BLE001
        return None

    predicted = normalize_category_output(raw, candidates)
    if predicted is None:
        return None

    return {
        "id": item.get("id"),
        "_index": index,
        "category": predicted,
        "gold_category": _gold_category(item),
        "candidate_categories": candidates,
        "task_type": trace,
    }


def topic_classify_placeholder(item: Dict[str, Any], index: int, reason: str) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "_index": index,
        "category": None,
        "gold_category": _gold_category(item),
        "candidate_categories": list(CATEGORIES),
        "task_type": f"failed:{reason}",
    }


def topic_classify_succeeded(row: Dict[str, Any]) -> bool:
    return isinstance(row.get("category"), str) and row["category"] in CATEGORIES


# --------------------------------------------------------------------------- #
# Direct baseline: 31-way classification without evidence extraction
# --------------------------------------------------------------------------- #
def build_baseline_prompt(item: Dict[str, Any], candidates: Sequence[str]) -> str:
    category_text = "\n".join(f"- {category}" for category in candidates)
    content = item.get("clause_content") or item.get("content", "") or ""
    return f"""你是一名中文法律法规条文主题分类助手。

任务：
根据法规标题和条文内容，判断该条文最主要归属的法规主题类别。

只能从以下候选类别中选择一个：
{category_text}

判断原则：
1. 以条文的核心调整对象和主要法律关系为准，不要只看单个关键词。
2. 法规标题可以辅助判断；标题与条文内容不一致时，以条文内容为准。
3. 一个条文涉及多个领域时，选择最直接、最具体、最能概括本条规范目的的类别。

输出要求：
只输出一个候选类别名称，不要解释、标点、JSON 或其他文字。

【法规标题】
{item.get("title", "") or ""}

【条文内容】
{content}
""".strip()


def baseline_row(
    backend: BaseBackend,
    item: Dict[str, Any],
    index: int,
) -> Optional[Dict[str, Any]]:
    candidates = list(CATEGORIES)
    prompt = build_baseline_prompt(item, candidates)

    system_prompt = "你是中文法律法规条文主题分类助手，只输出一个候选类别名称。"
    try:
        raw = backend.generate(system_prompt, prompt)
    except Exception:  # noqa: BLE001
        return None

    predicted = normalize_category_output(raw, candidates)
    if predicted is None:
        return None

    return {
        "id": item.get("id"),
        "_index": index,
        "category": predicted,
        "gold_category": _gold_category(item),
        "candidate_categories": candidates,
        "task_type": "direct",
    }


def baseline_placeholder(item: Dict[str, Any], index: int, reason: str) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "_index": index,
        "category": None,
        "gold_category": _gold_category(item),
        "candidate_categories": list(CATEGORIES),
        "task_type": f"direct_failed:{reason}",
    }


def baseline_succeeded(row: Dict[str, Any]) -> bool:
    return isinstance(row.get("category"), str) and row["category"] in CATEGORIES
