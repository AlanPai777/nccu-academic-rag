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
from rag.agent_runtime import _ALL_OFFICE_NAMES
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


# ── Structural checklist: form table → required stations (2026-08-26) ───────
# Generalizes eval.py's static nine_stamps/office_locations criteria (which
# hardcode 休學's 7-station chain into the synthesis prompt) into a
# dynamically-extracted checklist item, using the moltke form's OWN table
# structure instead of a procedure-specific rule. This is the "right signal"
# a reverted earlier attempt (office_role_diversity, see self_eval_node's
# comments) went looking for but used the wrong data source for — the real
# signal lives here, in the form table's own numbered-station markers, not
# in the office contact roster's job-title diversity.
#
# Matches both full-width（）and half-width() parens, and mixed pairs like
# "(7）" — confirmed present in real form text (QP-T01-03-02/QP-T01-03-05
# both mix conventions within the same table). Requires a digit directly
# inside the parens so it doesn't collide with _NOTE_MARKER_RE's "1、"-style
# markers (different syntax, already handled separately) or unrelated
# parenthetical text like "(未成年學生須經法定代理人簽章同意。)".
_STATION_MARKER_RE = re.compile(r'[（(]\s*(\d{1,2})\s*[）)]\s*([^\s|（(]{1,20})')


def _extract_stations(text: str) -> list[dict]:
    """
    Find numbered station markers (e.g. "（1）系所簽章") in one page's text,
    return [{"step": 1, "label": "系所簽章"}, ...] sorted by step number,
    deduplicated by step (first occurrence wins — a station number is
    sometimes referenced again later in the same table).
    """
    stations: dict[int, str] = {}
    for m in _STATION_MARKER_RE.finditer(text):
        step = int(m.group(1))
        label = m.group(2).strip()
        if label and step not in stations:
            stations[step] = label
    return [{"step": step, "label": stations[step]} for step in sorted(stations)]


def _normalize_station_label(label: str) -> str:
    """
    Map a raw station label to a canonical office name from
    _ALL_OFFICE_NAMES when one is embedded in it (e.g. "生僑組/原資中心" →
    "生僑組", "教務處註冊組" → "教務處") — the raw label is often a compound
    that the synthesized answer won't reproduce verbatim, but will almost
    always mention the canonical office name it contains. Falls back to the
    raw label unchanged when no canonical name matches (e.g. "系所簽章" has
    no office-name entry — kept as-is since it's already a short, matchable
    term on its own).
    """
    for name in _ALL_OFFICE_NAMES:
        if name in label:
            return name
    return label


def _extract_best_stations(context_pages: list[dict]) -> dict:
    """
    Take the FIRST moltke.nccu.edu.tw page (in context_pages order) with
    >=2 station markers — one coherent procedure's stations should all come
    from ONE page/form, not merged across pages (merging risks combining
    unrelated numbered lists into one garbled sequence). "First", not "most
    matches": context_pages is anchor-page(s)-first, expand-fetched
    cross-references after (retrieval_anchor_node's own fetch is written
    before retrieval_expand_node's Send-branch contributions land via the
    operator.add reducer) — the anchor page is the one grep_texts matched
    directly against the query's own topic, so it's the actually-relevant
    form. 2026-08-26 confirmed picking by raw count instead is wrong: for
    "如何辦理休學", QP-T01-03-02 (the correct, anchor-fetched 休學 form) has
    a clean 7-station table, but QP-T01-03-05 (退學's form — a DIFFERENT
    procedure, only pulled in because 休學's overview page cross-references
    it) has 7 stations too PLUS 1 noise match (the moltke viewer renders
    QP-T01-02-05's 委託書 content inline on other forms' pages, which this
    regex also matches) for a spurious total of 8 — outscoring and replacing
    the correct table under a "most matches" rule. "First" avoids this by
    construction, without needing to specifically pattern-match away every
    possible noise source.

    Requires >=2 stations to count as a real table, not a single stray
    "(3)" false-triggering; scoped to moltke.nccu.edu.tw pages (get_form()'s
    own domain) since station tables structurally only exist on official
    form documents, not general aca/osa pages (also confirmed 2026-08-26: a
    非-form PDF with a numbered 公文 reference list otherwise false-triggers
    this).
    """
    for page in context_pages:
        if "moltke.nccu.edu.tw" not in page.get("url", ""):
            continue
        stations = _extract_stations(page.get("text", ""))
        if len(stations) >= 2:
            return {
                "stations":     [{"step": s["step"], "label": _normalize_station_label(s["label"])} for s in stations],
                "source_title": page.get("title", ""),
            }
    return {"stations": [], "source_title": ""}


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

    station_info = _extract_best_stations(context_pages)

    checklist = {
        "person_names":    person_names,
        "forms":           forms,
        "notes":           notes,
        "stations":        station_info["stations"],
        "stations_source": station_info["source_title"],
    }
    print(
        f"[extraction] {len(person_names)} person(s), "
        f"forms={[(f['id'], f['title']) for f in forms]}, "
        f"notes={len(notes)}, stations={[s['label'] for s in station_info['stations']]}",
        file=sys.stderr,
    )
    return {**state, "extraction_checklist": checklist}


def _format_checklist(checklist: dict) -> str:
    """Format extraction_checklist into a prompt-injectable block, or "" if empty."""
    if not checklist:
        return ""

    lines = ["【本次搜尋結果中實際找到的事實 checklist——以下項目若跟問題相關就必須出現在答案中，"
              "不要遺漏；checklist 沒列出的人名/表單/審核層級也不要自行補上】"]

    if checklist.get("stations"):
        lines.append(f"表單記載的辦理站點順序（來源：{checklist.get('stations_source', '') or '（無標題）'}）——"
                     "答案的步驟清單必須涵蓋以下每一站，不要遺漏；也不要自行增加表單沒列出的站：")
        for s in checklist["stations"]:
            lines.append(f"  {s['step']}. {s['label']}")

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
