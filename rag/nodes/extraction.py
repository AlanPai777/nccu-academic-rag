"""
rag/nodes/extraction.py
extraction_node: condition 6, pre-synthesis dynamic checklist — builds a
fact checklist from what retrieval/office_lookup actually found, before
synthesis runs. Pure regex, zero extra LLM calls. _format_checklist() lives
here too since it's the direct consumer-facing counterpart of the checklist
this module builds (synthesis_node imports it from here).
"""

from __future__ import annotations

import re
import sys

from rag.agent_tools import extract_form_ids
from rag.nodes.state import AgentState

# Matches the numbering conventions Chinese-language forms use for
# "注意事項"/"說明"-style enumerated notes — Arabic-顿號 ("1、2、3、...",
# 休學's QP-T01-03-02), Chinese-numeral-顿號 ("一、二、三、...", 復學's
# QP-T01-03-04, which also nests an Arabic sub-list under one item), and
# Arabic-句點 ("1. 2. 3. ...", QP-T01-02-05's own "委託代辦說明" section —
# found 2026-08-26 while checking whether this regex generalized: it didn't,
# 0 notes extracted from that form until this separator was added).
# Deliberately a closed, enumerable set (unlike open-ended vocabularies such
# as office job titles) — Chinese formal documents only use a small number
# of numbering conventions, so widening this regex to cover them is not the
# same class of blind spot as _ADMIN_TITLES (office_lookup_skill.py); still,
# no claim this exhausts every convention (e.g. ①②③ or （一）（二） are not
# covered) — just the ones confirmed present in this corpus so far.
_NOTE_MARKER_RE = re.compile(r'(?:^|(?<=[。\s]))([一二三四五六七八九十]{1,3}|[0-9]{1,2})[、.]')
_TRAILING_TABLE_PIPES_RE = re.compile(r'(\s*\|)+\s*$')


def _extract_candidate_notes(text: str) -> list[str]:
    """
    Find "注意事項"-style enumerated notes and return each item as its own
    string. Scoped to ONE LINE at a time (`\\n`-split) rather than scanning
    the whole page/form text — 2026-08-26 lesson from a reverted earlier
    attempt: these notes live inside a single markdown table ROW (one
    source-line per row), sandwiched between unrelated rows before and
    after (e.g. QP-T01-03-02's 說明 row sits between a 住宿組/國際學生 row
    and a 申請人簽章/領取方式 row). Cutting from the first marker to the
    end of the whole text crossed those row boundaries and mangled both
    the notes and the unrelated rows around them. A line needs 2+ markers
    to count (avoids firing on a single stray "N、" inside ordinary prose,
    e.g. a line mentioning "7日內" without being a real enumerated list).
    """
    notes: list[str] = []
    for line in text.split("\n"):
        markers = list(_NOTE_MARKER_RE.finditer(line))
        if len(markers) < 2:
            continue
        for i, m in enumerate(markers):
            start = m.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(line)
            item = _TRAILING_TABLE_PIPES_RE.sub("", line[start:end]).strip()
            if item:
                notes.append(item)
    return notes


def extraction_node(state: AgentState) -> AgentState:
    """
    Condition 6: build a fact checklist from what retrieval/office_lookup
    actually found, BEFORE synthesis — v1 is pure regex, zero extra LLM calls.

    person_names: built directly from office_lookup_result's structured
                   {office: {contacts: [{name, duty, ext, email}, ...]}} —
                   reading OfficeLookupSkill's own per-office contact lists,
                   not regex re-parsing the flattened office_context string
                   (was `_PERSON_LINE_RE` matching "• 姓名（職責）分機 XXXXX"
                   lines — worked but was a fragile round-trip through
                   already-formatted text; office_lookup_result carries the
                   same data structured, so this reads it directly instead).
    forms:        form IDs parsed from context_pages text via agent_tools.extract_form_ids()
                   (reused, not reimplemented), each paired with its title when
                   the form's own page is in context_pages — lets the LLM judge
                   relevance instead of guessing from a bare ID.

    Replaces static content rules in _SYNTHESIS_PROMPT (old rules 3/5/7: fixed
    3-layer 教務處 review, fixed supplementary-form list, fixed 身心健康中心
    mention) with "here is what you found" — the LLM is told what's actually
    in the data instead of being told what to say regardless of the query.
    Directly targets the hallucination confirmed in Step 1 Q2 (復學 answer
    fabricated a 教務長 approval layer that doesn't exist in QP-T01-03-04).
    """
    person_names: list[dict] = []
    for office, info in state.get("office_lookup_result", {}).items():
        for c in info.get("contacts", []):
            name, duty, ext = c.get("name", ""), c.get("duty", ""), c.get("ext", "")
            if not name or not ext:
                continue
            person_names.append({"name": name, "duty": duty, "ext": ext})

    context_pages = state.get("context_pages", [])

    form_ids: set[str] = set()
    for page in context_pages:
        form_ids.update(extract_form_ids(page.get("text", "")))

    # Attach a title to each form ID when its own page is in context_pages
    # (true whenever ProcedureSkill actually called get_form() on it) — a bare
    # ID like "QP-T01-03-05" gives the LLM nothing to judge relevance by; the
    # title ("學生退學離校申請書") makes it obvious that one's a different
    # procedure (退學, not 休學) and should be left out, while a form whose
    # title clearly complements the question should be kept.
    forms: list[dict] = []
    for fid in sorted(form_ids):
        title = next(
            (p.get("title", "") for p in context_pages if fid in p.get("url", "")),
            "",
        )
        forms.append({"id": fid, "title": title})

    # Tagged with source_title (not flattened into one pool) so the prompt
    # can present notes grouped by which page/form they came from — 2026-08-26
    # finding: a straight flat list mixed 休學's own 6-8 notes with 7 more
    # from 復學's QP-T01-03-04 (cross-referenced on the same aca overview
    # page, fetched by expand alongside 休學's own form), diluting the
    # candidate pool with a different procedure's content. Reuses the SAME
    # title-based relevance judgment _SYNTHESIS_PROMPT already asks the LLM
    # to apply to `forms` below, instead of inventing a new filter.
    notes: list[dict] = []
    seen_notes: set[str] = set()
    for page in context_pages:
        source_title = page.get("title", "")
        for note in _extract_candidate_notes(page.get("text", "")):
            if note not in seen_notes:
                seen_notes.add(note)
                notes.append({"text": note, "source_title": source_title})

    checklist = {
        "person_names": person_names,
        "forms":        forms,
        "notes":        notes,
    }
    print(
        f"[extraction] {len(person_names)} person(s), "
        f"forms={[(f['id'], f['title']) for f in forms]}, "
        f"notes={len(notes)}",
        file=sys.stderr,
    )
    return {**state, "extraction_checklist": checklist}


def _format_checklist(checklist: dict) -> str:
    """Format extraction_checklist into a prompt-injectable block, or "" if empty."""
    if not checklist:
        return ""

    lines = ["【本次搜尋結果中實際找到的事實 checklist——以下項目若跟問題相關就必須出現在答案中，"
              "不要遺漏；checklist 沒列出的人名/表單/審核層級也不要自行補上】"]

    if checklist.get("person_names"):
        lines.append("承辦人：")
        for p in checklist["person_names"]:
            lines.append(f"  • {p['name']}（{p['duty']}）分機 {p['ext']}")

    if checklist.get("forms"):
        lines.append("相關表單（逐一判斷是否與學生問題相關；標題明顯是同一流程或直接後續步驟的，"
                     "必須在答案中列出，不可省略；標題明顯是不同流程的可以略過）：")
        for f in checklist["forms"]:
            title = f"（{f['title']}）" if f["title"] else ""
            lines.append(f"  • {f['id']}{title}")

    if checklist.get("notes"):
        lines.append("表單/頁面列出的補充說明事項（按來源分組——判斷相關性的方式跟上面「相關表單」規則一樣："
                     "來源標題明顯是同一流程的，其補充說明若與學生情況相關就適當納入答案；"
                     "來源標題明顯是不同流程的，整組可以略過，不需要每一則都寫進答案）：")
        last_source = None
        for n in checklist["notes"]:
            if n["source_title"] != last_source:
                lines.append(f"  [來源：{n['source_title'] or '（無標題）'}]")
                last_source = n["source_title"]
            lines.append(f"    • {n['text']}")

    if len(lines) == 1:
        return ""  # nothing extracted — omit the section entirely
    return "\n".join(lines) + "\n\n"
