from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
import re
from pathlib import Path
from typing import Iterable, Literal
from urllib.parse import unquote, urlparse

import requests
from dotenv import load_dotenv
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader
from tavily import TavilyClient
from typing_extensions import Annotated, TypedDict


SYSTEM_PROMPT = """You are a careful research assistant.

Your job is to answer research and analysis questions with clear, structured, educational explanations.
You operate fully autonomously toward the user's stated research goal.

Autonomy rules:
- Never ask the user follow-up questions.
- Make reasonable assumptions when details are missing and explicitly list those assumptions.
- Use available tools proactively until the goal is complete or evidence is sufficient.
- For any web-search-related or latest-information task, always call tavily_web_search.
- If Tavily results include PDF URLs, call download_pdfs to save them locally.
- After downloading PDFs, call parse_pdfs_for_sections to extract key structured content (abstract, methods, findings).
- Use extracted PDF sections to synthesize high-quality responses.
- Cite URLs and paper titles in your final answer.
- End every assistant message with exactly one of these lines:
    GOAL_STATUS: IN_PROGRESS
    GOAL_STATUS: COMPLETE

Completion criteria:
- Mark COMPLETE only when you provide a final synthesis that addresses the user's requested objective.
- If you cannot complete due to missing public evidence, explain constraints and still mark COMPLETE.

When the topic is health related:
- Give general educational information.
- Call out uncertainty when appropriate.
- Avoid presenting the answer as medical diagnosis or personal medical advice.
- Suggest consulting a qualified clinician for individual medical concerns.
"""

HEALTH_RESEARCH_EXAMPLES = [
    "How does our body digest protein from the moment we eat it to the point where amino acids are used by cells?",
    "Research goal: evaluate whether high-protein diets improve satiety and body composition in adults. Scope: studies from 2019 onward, prioritize peer-reviewed human studies and meta-analyses. Deliverable: concise evidence summary, conflicting findings, and practical takeaways with source links.",
    "Research goal: compare policy approaches for reducing urban air pollution in megacities. Scope: evidence from government reports and peer-reviewed studies from the last 5 years. Deliverable: strategy comparison table, tradeoffs, and recommended roadmap with citations.",
]

AUTONOMY_NUDGE = (
    "Continue autonomously toward the research goal. Use tools as needed. "
    "Do not ask the user questions. If the objective is fully addressed, deliver the final report "
    "and end with GOAL_STATUS: COMPLETE."
)

CREDIBLE_DOMAIN_HINTS = (
    ".gov",
    ".edu",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "nature.com",
    "thelancet.com",
    "bmj.com",
    "who.int",
    "sciencedirect.com",
    "nejm.org",
    "jamanetwork.com",
    "pnas.org",
    "frontiersin.org",
    "springer.com",
    "wiley.com",
    "cell.com",
    "oup.com",
    "mdpi.com",
    "plos.org",
    "researchgate.net",
    "arxiv.org",    
)


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    goal_complete: bool


def build_model() -> AzureChatOpenAI:
    load_dotenv()

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is not set. Add it to your environment or .env file."
        )

    model_name = os.getenv("RESEARCH_AGENT_MODEL", "gpt-4o")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    return AzureChatOpenAI(
        azure_deployment=model_name,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=0,
    )


def _tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to your environment or .env file."
        )
    return TavilyClient(api_key=api_key)


def _is_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return path.endswith(".pdf") or ".pdf" in path


def _looks_credible(url: str) -> bool:
    lowered = url.lower()
    return any(hint in lowered for hint in CREDIBLE_DOMAIN_HINTS)


def _goal_complete_from_ai(content: object) -> bool:
    text = str(content).upper()
    return "GOAL_STATUS: COMPLETE" in text


def _max_autonomous_steps() -> int:
    raw = os.getenv("MAX_AUTONOMOUS_STEPS", "10")
    try:
        parsed = int(raw)
    except ValueError:
        return 10
    return max(parsed, 1)


@tool
def tavily_web_search(query: str, max_results: int = 10) -> str:
    """Search the web for the latest credible research sources and include PDF links when available."""

    client = _tavily_client()
    enhanced_query = (
        f"{query} latest research papers peer-reviewed PDF site:gov OR site:edu OR pubmed"
    )
    response = client.search(query=enhanced_query, max_results=max_results)

    results = response.get("results", [])
    formatted_results: list[dict[str, str]] = []
    pdf_urls: list[str] = []

    # pdf_urls.append("https://pmc.ncbi.nlm.nih.gov/articles/PMC12145679/pdf/main.pdf")

    for item in results:
        url = item.get("url", "")
        title = item.get("title", "Untitled")
        content = item.get("content", "")

        if _is_pdf_url(url):
            pdf_urls.append(url)

        formatted_results.append(
            {
                "title": title,
                "url": url,
                "snippet": content,
                "credible_source": "yes" if _looks_credible(url) else "unknown",
                "is_pdf": "yes" if _is_pdf_url(url) else "no",
            }
        )

    payload = {
        "original_query": query,
        "enhanced_query": enhanced_query,
        "result_count": len(formatted_results),
        "pdf_urls": sorted(set(pdf_urls)),
        "results": formatted_results,
    }
    return json.dumps(payload, indent=2)


@tool
def download_pdfs(pdf_urls: list[str], output_dir: str = "papers") -> str:
    """Download PDF URLs into a local folder and return a summary of saved files."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []

    for url in pdf_urls:
        try:
            if not url.lower().startswith(("http://", "https://")):
                raise ValueError("URL must start with http:// or https://")
            if not _is_pdf_url(url):
                raise ValueError("URL does not appear to reference a PDF")

            parsed = urlparse(url)
            name = Path(unquote(parsed.path)).name or "paper.pdf"
            if not name.lower().endswith(".pdf"):
                name = f"{name}.pdf"

            digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
            safe_name = f"{Path(name).stem}_{digest}.pdf"
            destination = target_dir / safe_name

            with requests.get(url, stream=True, timeout=45) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if "pdf" not in content_type and not _is_pdf_url(url):
                    raise ValueError(
                        f"Downloaded content is not PDF-like (Content-Type: {content_type})"
                    )

                with open(destination, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            downloaded.append({"url": url, "saved_to": str(destination)})
        except Exception as exc:
            failed.append({"url": url, "error": str(exc)})

    payload = {
        "output_dir": str(target_dir),
        "requested": len(pdf_urls),
        "downloaded": downloaded,
        "failed": failed,
    }
    return json.dumps(payload, indent=2)


def _extract_section(text: str, section_name: str) -> str:
    """Extract a section from PDF text using regex patterns."""
    # Common section headers with case-insensitive matching
    section_patterns = {
        "abstract": r"(?:^|\n)(?:ABSTRACT|Abstract)(?:\s|\n)([\s\S]*?)(?=\n(?:INTRODUCTION|INTRODUCTION|METHODS|1\.|\d\.\s|$))",
        "methods": r"(?:^|\n)(?:METHODS|Methods)(?:\s|\n)([\s\S]*?)(?=\n(?:RESULTS|DISCUSSION|FINDINGS|CONCLUSION|\d\.\s|$))",
        "findings": r"(?:^|\n)(?:FINDINGS|RESULTS|Findings|Results)(?:\s|\n)([\s\S]*?)(?=\n(?:DISCUSSION|CONCLUSION|LIMITATIONS|REFERENCES|\d\.\s|$))",
        "discussion": r"(?:^|\n)(?:DISCUSSION|Discussion)(?:\s|\n)([\s\S]*?)(?=\n(?:CONCLUSION|REFERENCES|LIMITATIONS|$))",
        "conclusion": r"(?:^|\n)(?:CONCLUSION|CONCLUSIONS|Conclusion)(?:\s|\n)([\s\S]*?)(?=\n(?:REFERENCES|ACKNOWLEDGMENTS|$))",
    }
    
    pattern = section_patterns.get(section_name.lower())
    if not pattern:
        return ""
    
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
        # Truncate to first 1500 chars to avoid token bloat
        return extracted[:1500] if len(extracted) > 1500 else extracted
    return ""


@tool
def parse_pdfs_for_sections(output_dir: str = "papers") -> str:
    """Parse downloaded PDFs and extract structured sections (abstract, methods, findings, discussion)."""
    
    target_dir = Path(output_dir)
    if not target_dir.exists():
        return json.dumps({
            "output_dir": str(target_dir),
            "pdfs_found": 0,
            "extracted": [],
            "errors": ["Papers directory does not exist yet. Download PDFs first."],
        }, indent=2)
    
    pdf_files = sorted(target_dir.glob("*.pdf"))
    if not pdf_files:
        return json.dumps({
            "output_dir": str(target_dir),
            "pdfs_found": 0,
            "extracted": [],
            "errors": ["No PDF files found in papers directory."],
        }, indent=2)
    
    extracted: list[dict] = []
    errors: list[str] = []
    
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            if len(reader.pages) == 0:
                errors.append(f"{pdf_path.name}: PDF has no pages.")
                continue
            
            # Extract text from first 10 pages (balance quality vs token count)
            text = ""
            for page_num in range(min(10, len(reader.pages))):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
            
            if not text.strip():
                errors.append(f"{pdf_path.name}: Could not extract text.")
                continue
            
            sections = {
                "abstract": _extract_section(text, "abstract"),
                "methods": _extract_section(text, "methods"),
                "findings": _extract_section(text, "findings"),
                "discussion": _extract_section(text, "discussion"),
                "conclusion": _extract_section(text, "conclusion"),
            }
            
            # Only include sections that have content
            sections = {k: v for k, v in sections.items() if v}
            
            extracted.append({
                "filename": pdf_path.name,
                "pages_processed": min(10, len(reader.pages)),
                "total_pages": len(reader.pages),
                "sections": sections,
            })
        except Exception as exc:
            errors.append(f"{pdf_path.name}: {str(exc)}")
    
    payload = {
        "output_dir": str(target_dir),
        "pdfs_found": len(pdf_files),
        "successfully_parsed": len(extracted),
        "extracted": extracted,
        "errors": errors,
    }
    return json.dumps(payload, indent=2)


def build_agent():
    model = build_model()

    tools = [tavily_web_search, download_pdfs, parse_pdfs_for_sections]
    tools_by_name = {tool_item.name: tool_item for tool_item in tools}
    model_with_tools = model.bind_tools(tools)

    def llm_call(state: ResearchState) -> ResearchState:
        response = model_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        )
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
            "goal_complete": _goal_complete_from_ai(response.content),
        }

    def tool_node(state: ResearchState) -> ResearchState:
        tool_messages: list[ToolMessage] = []
        for tool_call in state["messages"][-1].tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]
            observation = selected_tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=observation, tool_call_id=tool_call["id"])
            )
        return {
            "messages": tool_messages,
            "llm_calls": state.get("llm_calls", 0),
            "goal_complete": state.get("goal_complete", False),
        }

    def autonomy_nudge(state: ResearchState) -> ResearchState:
        return {
            "messages": [HumanMessage(content=AUTONOMY_NUDGE)],
            "llm_calls": state.get("llm_calls", 0),
            "goal_complete": state.get("goal_complete", False),
        }

    def should_continue(
        state: ResearchState,
    ) -> Literal["tool_node", "autonomy_nudge", "__end__"]:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tool_node"
        if state.get("goal_complete", False):
            return END
        if state.get("llm_calls", 0) >= _max_autonomous_steps():
            return END
        return "autonomy_nudge"

    graph = StateGraph(ResearchState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("autonomy_nudge", autonomy_nudge)
    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges(
        "llm_call", should_continue, ["tool_node", "autonomy_nudge", END]
    )
    graph.add_edge("tool_node", "llm_call")
    graph.add_edge("autonomy_nudge", "llm_call")
    return graph.compile()


def run_query(query: str) -> dict:
    agent = build_agent()
    return agent.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "llm_calls": 0,
            "goal_complete": False,
        }
    )


def _format_ai_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def print_response(result: dict) -> None:
    final_message = result["messages"][-1]
    if isinstance(final_message, AIMessage):
        print(_format_ai_content(final_message.content))
        print(f"\nLLM calls: {result['llm_calls']}")
        if not result.get("goal_complete", False) and result.get("llm_calls", 0) >= _max_autonomous_steps():
            print(
                "Reached MAX_AUTONOMOUS_STEPS before explicit completion. "
                "Increase MAX_AUTONOMOUS_STEPS if you want longer autonomous execution."
            )
        return

    print(final_message)


def show_examples(examples: Iterable[str]) -> None:
    for index, example in enumerate(examples, start=1):
        print(f"{index}. {example}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an autonomous LangGraph research agent with Tavily web search, "
            "PDF download, and PDF parsing tools for structured research synthesis."
        )
    )
    parser.add_argument(
        "query",
        nargs="?",
        help=(
            "Research goal prompt. Provide full objective, scope, and desired output in this first input."
        ),
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Print built-in health research example prompts.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.examples:
        show_examples(HEALTH_RESEARCH_EXAMPLES)
        return

    if not args.query:
        raise SystemExit(
            "Provide a query or run with --examples to view the built-in research prompts."
        )

    result = run_query(args.query)
    print_response(result)


if __name__ == "__main__":
    main()
