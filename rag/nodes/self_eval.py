"""
rag/nodes/self_eval.py
self_eval_node (E4): self-evaluation checklist for PROCEDURE answers, plus
its supporting helpers — the FullCite-inspired posthoc note-similarity check
and condition 8-C's Router-as-judge semantic check.
"""

from __future__ import annotations

import json
import re
import sys

from rag.router import QueryType
from rag.llm_client import simple_chat
from rag.domain_router import is_ambiguous
from rag.nodes.state import AgentState

_MAX_SELF_EVAL_RETRIES = 2

# person_names dropped from here — condition 6 made it dynamic (see self_eval_node
# below): it now checks against extraction_checklist's actual person_names for
# THIS query instead of a fixed list of 6 休學-specific names hardcoded regardless
# of query type. sources/procedure_format stay static — they're format checks,
# not content specific to any one procedure.
def _char_bigrams(s: str) -> set[str]:
    s = re.sub(r'\s+', '', s)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}


def _note_referenced(note: str, answer: str, threshold: float = 0.4) -> bool:
    """
    Posthoc similarity check (FullCite-inspired, 2026-08-26) for whether a
    checklist note's content shows up somewhere in the answer — used
    instead of requiring the LLM to emit a rigid citation ID (which
    "Models Can Model, But Can't Bind" and this session's own experience
    both suggest is an unreliable ask of an untrained model) or requiring
    literal substring match (too strict — the LLM is expected to
    paraphrase/summarize a note, not quote the source verbatim). Character
    bigram overlap coefficient (bigrams(note) ∩ bigrams(answer) /
    bigrams(note)) rather than full Jaccard, since a short note compared
    against the whole (much longer) answer via symmetric Jaccard would
    always score near-zero regardless of match quality. Threshold 0.4
    empirically separates a real paraphrase (~0.76 on a real 休學 note) from
    unrelated answer text (~0.06) with margin to spare.
    """
    note_bg = _char_bigrams(note)
    if not note_bg:
        return False
    answer_bg = _char_bigrams(answer)
    return len(note_bg & answer_bg) / len(note_bg) >= threshold


# ── Condition 8-C (Step 6): Router-as-judge semantic check ──────────────────
# Standalone helper, NOT yet wired into self_eval_node's retry loop — keyword
# checks (_SELF_EVAL_CRITERIA below) can only verify certain strings appear,
# they structurally cannot catch "format right, content off-topic" answers
# (e.g. Domain Router's osa-dilution finding: a 復學 answer that's actually
# mostly about dorm eligibility, but still contains "復學" and step-list
# formatting so every keyword check passes). Known weakness from literature
# (Zheng et al. 2023, MT-Bench "LLM-as-judge"): the judge itself can be
# wrong or format-sensitive — validated here against curated known-good/
# known-bad examples BEFORE deciding whether to wire it into the live retry
# loop, not assumed reliable by default.
_ROUTER_JUDGE_SYSTEM = "你是 NCCU 學術事務 Q&A 系統的品質判官，只根據語意判斷答案內容是否真的回答了問題，不看格式對不對。"

_ROUTER_JUDGE_USER = """學生問題：{query}

系統答案：
{answer}

請判斷：這個答案的內容是否真的針對「{query}」這個問題本身作答？特別留意「格式正確、關鍵字也對，但內容其實主要在講別的主題」這種情況——例如答案主要在介紹某個不相關的規定或活動，只是剛好提到問題裡的關鍵字或表面相關。

只回傳一個 JSON，不要有其他文字：{{"verdict": "PASS 或 FAIL", "reason": "一句話原因"}}"""


def _router_judge(query: str, answer: str) -> dict:
    """Send (query, answer) to the LLM for a semantic PASS/FAIL judgment."""
    raw = simple_chat(
        messages=[
            {"role": "system", "content": _ROUTER_JUDGE_SYSTEM},
            {"role": "user",   "content": _ROUTER_JUDGE_USER.format(query=query, answer=answer)},
        ],
        max_tokens=200,
    )
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"verdict": "ERROR", "reason": raw[:200]}


_SELF_EVAL_CRITERIA = [
    {
        "name":     "sources",
        "keywords": ["引用來源", "https://"],
        "min_hits": 1,
        "hint":     "答案缺少引用來源。請在最後加上【引用來源】並列出所有 URL。",
    },
    {
        "name":     "procedure_format",
        "keywords": ["步驟", "第一步", "①", "第1步", "1.", "1、"],
        "min_hits": 1,
        "hint":     "答案未使用步驟清單格式。請改為以編號步驟列出辦理流程。",
    },
]


def self_eval_node(state: AgentState) -> AgentState:
    """
    E4: Self-evaluation checklist for PROCEDURE answers.

    Checks answer against critical criteria.  If any check fails and the
    retry budget has not been exhausted, sets correction_hint and increments
    iteration so that _after_self_eval routes back to synthesis_node.
    """
    # Only evaluate PROCEDURE answers; KNOWLEDGE/CONTACT pass through.
    if state["query_type"] != QueryType.PROCEDURE:
        return state

    iteration = state.get("iteration", 0)
    if iteration >= _MAX_SELF_EVAL_RETRIES:
        return state  # budget exhausted — accept current answer

    answer  = state.get("answer", "")
    failures = []

    for check in _SELF_EVAL_CRITERIA:
        hits = sum(1 for kw in check["keywords"] if kw in answer)
        if hits < check["min_hits"]:
            failures.append(check["hint"])

    # Condition 6: dynamic person_names check against this query's own
    # extraction_checklist instead of a hardcoded 休學-specific name list.
    checklist_names = [p["name"] for p in state.get("extraction_checklist", {}).get("person_names", [])]
    if checklist_names:
        required = min(2, len(checklist_names))
        hits = sum(1 for name in checklist_names if name in answer)
        if hits < required:
            failures.append(
                f"答案缺少承辦人姓名（需 ≥{required} 位，checklist 中找到：{'、'.join(checklist_names)}）。"
                "請在每個步驟列出對應承辦人姓名（從【各辦公室聯絡資訊】引用）。"
            )

    # 2026-08-26: tried a dynamic check here — "does the answer show
    # multiple review layers for offices whose real contact roster has 2+
    # distinct job titles" — as a generalization of eval.py's static
    # nine_stamps criterion. Reverted: distinct-duty-count in
    # OfficeLookupSkill's contact roster measures staffing diversity (how
    # many different job titles an office happens to employ — 住宿組 has 13,
    # purely from having lots of staff with different day-to-day jobs), not
    # documented approval-chain depth. That's the wrong data source — it
    # would force 2+ named contacts onto offices that don't need them,
    # recreating the original roster-bloat bug for offices other than the
    # one it was meant to fix. The right signal (whether the PROCEDURE
    # ITSELF documents multiple approval sub-stages for a step — e.g.
    # QP-T01-03-02's table literally splits 教務處's row into 組長 and
    # 教務長 sub-cells) lives in context_pages' table structure, not in the
    # office contact roster; extracting that is a separate, harder task,
    # not done yet. See phase_f_planning_report.md Gotchas.

    # mental_health_note (condition 6 extension, 2026-08-26): dynamic check
    # against extraction_checklist's own "notes" — enumerated 注意事項 items
    # regex-extracted from the actual form/page text (see
    # _extract_candidate_notes), not any procedure-specific hardcoded
    # phrase. Posthoc bigram-overlap matching (_note_referenced) instead of
    # requiring exact substring or a citation ID, since the LLM is expected
    # to paraphrase a note, not quote it verbatim. Same min(2, len()) floor
    # as person_names — not "all notes must appear" (some may genuinely be
    # irrelevant to this student's situation; forcing all of them in would
    # risk the same over-inclusion problem office_role_diversity just hit).
    checklist_notes = state.get("extraction_checklist", {}).get("notes", [])
    if checklist_notes:
        required = min(2, len(checklist_notes))
        hits = sum(1 for n in checklist_notes if _note_referenced(n["text"], answer))
        if hits < required:
            unreferenced = [n["text"] for n in checklist_notes if not _note_referenced(n["text"], answer)]
            preview = "；".join(n[:40] + ("…" if len(n) > 40 else "") for n in unreferenced[:3])
            failures.append(
                f"答案未充分反映表單的補充說明事項（需至少 {required} 則有對應到答案內容，"
                f"目前只有 {hits} 則；例如：{preview}）。"
                "請檢視這些補充說明是否與學生情況相關，相關的請適當納入答案（可用自己的話摘要，不需逐字照抄）。"
            )

    # Condition 8-C (Step 6): Router-as-judge, gated on domain_router's
    # is_ambiguous() so the extra LLM call only happens on queries where
    # routing was shaky to begin with (e.g. "如何辦理復學" — the real
    # osa-dilution case this was built for), not on every PROCEDURE answer.
    # On FAIL, the correction explicitly asks for an honest decline rather
    # than "try again" — re-synthesizing from the SAME context_pages can't
    # fix a wrong-subdomain retrieval, so asking for a rewrite would just
    # produce another confidently-off-topic answer. This is a stopgap, not
    # a fix for the underlying routing gap — see phase_f_planning_report.md
    # (Step 5's risk 2 / future Layer 3 CRAG-lite retry is the real fix).
    if is_ambiguous(state["query"]):
        judge = _router_judge(state["query"], answer)
        print(f"[router-judge] {judge.get('verdict')}: {judge.get('reason', '')}", file=sys.stderr)
        if judge.get("verdict") == "FAIL":
            failures.append(
                f"Router-as-judge 判定答案可能離題（{judge.get('reason', '')}）。"
                "若上方【搜尋到的頁面內容】明顯與問題主題不符，請不要勉強套用不相關的資料組出答案，"
                "改為誠實回答「資料不足以確認」並說明需要進一步查證。"
            )

    if not failures:
        return state  # all checks passed

    correction = (
        "【自評回饋：以下項目不符合要求，請重新生成更完整的答案】\n"
        + "\n".join(f"- {f}" for f in failures)
    )
    return {**state, "correction_hint": correction, "iteration": iteration + 1}
