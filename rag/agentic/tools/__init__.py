from rag.agentic.tools.search import search_texts
from rag.agentic.tools.grep import grep_texts_tool
from rag.agentic.tools.page import get_page_tool, extract_links_tool
from rag.agentic.tools.form import get_form_tool

TOOLS = [search_texts, grep_texts_tool, get_page_tool, extract_links_tool, get_form_tool]
