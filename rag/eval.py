"""
rag/eval.py
Auto-scoring framework for 休學辦理流程 Q&A evaluation.
13 criteria × 2 points = 26 max.

Changes from v1 (24 pts):
  - seven_stops  → nine_stamps  : 正確反映 QP-T01-03-02 表單的 9 個蓋章欄位；
                                   keywords 加入「組長」「教務長」；min_hits 提升至 7。
  - dean_approvals (NEW)        : 教務處內部需經組長批示 + 教務長核准兩層，明確驗證。
  - refund_table   min_hits 2→3 : 全額/2/3/1/3 三段均需出現，避免只提「退費」就得分。
  - contact_info   min_hits 2→3 : 要求涵蓋出納+教務+至少一個住宿/生僑/國合，共 3 支分機。
  - contact_persons: 加入「林啓屏」（教務長）；min_hits 1→2（需跨單位至少 2 人）。
  - pickup_method  min_hits 1→2 : 需同時提及「三個工作天」等時效 + 至少一種取件管道。
  - mental_health_note min_hits 1→2: 需同時提及身心健康中心名稱 + 電話或懷孕情境。
"""

EVAL_CRITERIA = [
    {
        "name":        "process_first",
        "description": "以步驟清單格式回答，不以規定開頭",
        "keywords":    ["步驟", "第一步", "①", "第1步", "流程"],
        "min_hits":    1,
    },
    {
        "name":        "nine_stamps",
        "description": "9 個蓋章欄位完整：前 6 站（系所/圖書館/出納/住宿/生僑/國合）+ 教務處 3 層（承辦/組長/教務長）",
        "keywords":    ["系所", "圖書館", "出納", "住宿組", "生僑", "國際合作", "教務", "組長", "教務長"],
        # min_hits=9：9 個關鍵字全部出現才給滿分。
        # 舊值 7 的 bug：只需 7 個地點關鍵字（系所…教務）就能達標，
        # 不需要「組長」「教務長」——但這兩欄正是 9 欄與 7 站的關鍵差異。
        "min_hits":    9,
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
        "name":        "dean_approvals",
        "description": "教務處需經組長批示及教務長核准（QP-T01-03-02 表單明列兩層核准）",
        "keywords":    ["組長", "教務長", "王揚忠", "林啓屏"],
        "min_hits":    2,
    },
    {
        "name":        "refund_table",
        "description": "退費比例表存在（全額退費 + 2/3 + 1/3 三欄均出現；接受中文或 Arabic 分數格式）",
        "keywords":    ["全額退費", "2/3", "1/3", "三分之二", "三分之一", "退費"],
        "min_hits":    3,
    },
    {
        "name":        "contact_info",
        "description": "至少 3 個電話分機（出納 62123 + 教務 63279 + 住宿/生僑/國合至少一處）",
        "keywords":    ["62123", "63279", "62222", "63251", "62040", "分機"],
        "min_hits":    3,
    },
    {
        "name":        "form_links",
        "description": "QP-T01-03-02 休學申請書連結存在（必要）",
        "keywords":    ["QP-T01-03-02"],
        "min_hits":    1,
    },
    {
        "name":        "supplementary_forms",
        "description": "補充表單：QP-T01-02-05 委託書 或 QP-T01-03-04 提早復學",
        "keywords":    ["QP-T01-02-05", "QP-T01-03-04"],
        "min_hits":    1,
    },
    {
        "name":        "no_hallucination",
        "description": "無幻覺（人工確認，自動給滿分）",
        "keywords":    [],
        "min_hits":    0,
    },
    {
        "name":        "contact_persons",
        "description": "至少提及 2 位承辦人姓名（需涵蓋不同單位）",
        "keywords":    ["盧誼甄", "王揚忠", "黃婉綝", "陳哲良", "林啓屏"],
        "min_hits":    2,
    },
    {
        "name":        "pickup_method",
        "description": "說明核准後至少 2 種領件方式（時效 + 管道）",
        "keywords":    ["三個工作天", "郵寄", "iNCCU", "領取"],
        "min_hits":    2,
    },
    {
        "name":        "mental_health_note",
        "description": "提及身心健康中心且附電話（8237-7419）或懷孕情境（至少 2 項）",
        "keywords":    ["身心健康中心", "8237-7419", "懷孕"],
        "min_hits":    2,
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
        "max":     26,
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
    電話分機：62123、63279、62222
    表單：QP-T01-03-02
    組長批示後送教務長核准
    王揚忠 林啓屏
    三個工作天後可至教務處領取或郵寄
    身心健康中心 8237-7419
    """
    print_score_report(sample)
