"""
rag/eval.py
Auto-scoring framework for eval_baseline_01_休學_zh.md.
8 criteria × 2 points = 16 max.
"""

EVAL_CRITERIA = [
    {
        "name":        "process_first",
        "description": "以步驟清單格式回答，不以規定開頭",
        "keywords":    ["步驟", "第一步", "①", "第1步", "流程"],
        "min_hits":    1,
    },
    {
        "name":        "seven_stops",
        "description": "7 站蓋章流程完整",
        "keywords":    ["系所", "圖書館", "出納", "住宿組", "生僑", "國際合作", "教務"],
        "min_hits":    5,
    },
    {
        "name":        "office_locations",
        "description": "每站有辦公室地點（行政大樓 X 樓）",
        "keywords":    ["行政大樓", "5樓", "3樓", "4樓", "8樓", "5 樓", "3 樓", "4 樓", "8 樓"],
        "min_hits":    3,
    },
    {
        "name":        "conditional_stops",
        "description": "住宿生 / 國際學生條件標注",
        "keywords":    ["住宿生", "國際學生"],
        "min_hits":    2,
    },
    {
        "name":        "refund_table",
        "description": "退費比例表存在",
        "keywords":    ["全額退費", "2/3", "1/3", "退費"],
        "min_hits":    2,
    },
    {
        "name":        "contact_info",
        "description": "至少 2 個電話分機號碼",
        "keywords":    ["62123", "63279", "62222", "63251", "分機"],
        "min_hits":    2,
    },
    {
        "name":        "form_links",
        "description": "QP-T01-03-02 連結存在",
        "keywords":    ["QP-T01-03-02"],
        "min_hits":    1,
    },
    {
        "name":        "no_hallucination",
        "description": "無幻覺（人工確認，自動給滿分）",
        "keywords":    [],
        "min_hits":    0,
    },
]


def score_answer(answer: str) -> dict:
    scores = {}
    total  = 0
    for c in EVAL_CRITERIA:
        if not c["keywords"]:
            score = 2  # no_hallucination: auto full marks, manual verification
        else:
            hits  = sum(1 for kw in c["keywords"] if kw in answer)
            score = 2 if hits >= c["min_hits"] else (1 if hits > 0 else 0)
        scores[c["name"]] = score
        total += score

    return {
        "scores":  scores,
        "total":   total,
        "max":     16,
        "missing": [k for k, v in scores.items() if v < 2],
    }


def print_score_report(answer: str) -> None:
    result = score_answer(answer)
    print(f"\n{'='*50}")
    print(f"總分：{result['total']}/{result['max']}")
    print(f"{'='*50}")
    for name, score in result["scores"].items():
        status = "✅" if score == 2 else ("⚠️" if score == 1 else "❌")
        print(f"  {status} {name}: {score}/2")
    if result["missing"]:
        print(f"\n缺少：{result['missing']}")


if __name__ == "__main__":
    sample = """
    步驟一：向系所辦公室提出申請
    第一步：確認申請時間...
    行政大樓3樓 出納組
    行政大樓5樓 住宿組（住宿生）
    國際學生請至8樓國際合作處
    退費標準：全額退費 / 2/3 / 1/3
    電話分機：62123、63279
    表單：QP-T01-03-02
    """
    print_score_report(sample)
