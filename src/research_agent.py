from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
from pathlib import Path
from typing import Iterable, Literal
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import requests
from dotenv import load_dotenv
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient
from typing_extensions import Annotated, TypedDict


SYSTEM_PROMPT = """You are a careful research assistant.

Your job is to answer research and analysis questions with clear, structured, educational explanations.
When the topic is health related:
- Give general educational information.
- Call out uncertainty when appropriate.
- Avoid presenting the answer as medical diagnosis or personal medical advice.
- Suggest consulting a qualified clinician for individual medical concerns.

Tool rules:
- For any web-search-related or latest-information task, always call tavily_web_search.
- If the Tavily results include PDF URLs, call download_pdfs to save them to the local papers folder.
- Base your final answer on the gathered sources and cite URLs in your response.
"""

HEALTH_RESEARCH_EXAMPLES = [
    "How does our body digest protein from the moment we eat it to the point where amino acids are used by cells?",
    "Find the latest credible research papers and PDF sources on protein digestion in humans, download the PDFs, and summarize key findings.",
    "Search recent peer-reviewed articles about sleep quality and muscle protein synthesis, then summarize the evidence.",
]

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

            headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    )
            }

            # response = requests.get(url, timeout=45)
            # response.raise_for_status()

            with requests.get(url, stream=True) as response:
                response.raise_for_status()

                with open(destination, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)


            content_type = response.headers.get("Content-Type", "").lower()

            if "pdf" not in content_type and not _is_pdf_url(url):
                raise ValueError(
                    f"Downloaded content is not PDF-like (Content-Type: {content_type})"
                )

            destination.write_bytes(response.content)
            
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


def build_agent():
    model = build_model()

    tools = [tavily_web_search, download_pdfs]
    tools_by_name = {tool_item.name: tool_item for tool_item in tools}
    model_with_tools = model.bind_tools(tools)

    def llm_call(state: ResearchState) -> ResearchState:
        response = model_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        )
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: ResearchState) -> ResearchState:
        tool_messages: list[ToolMessage] = []
        for tool_call in state["messages"][-1].tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]
            observation = selected_tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=observation, tool_call_id=tool_call["id"])
            )
        return {"messages": tool_messages, "llm_calls": state.get("llm_calls", 0)}

    def should_continue(state: ResearchState) -> Literal["tool_node", "__end__"]:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tool_node"
        return END

    graph = StateGraph(ResearchState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph.add_edge("tool_node", "llm_call")
    return graph.compile()


def run_query(query: str) -> dict:
    agent = build_agent()
    return agent.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "llm_calls": 0,
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
        return

    print(final_message)


def show_examples(examples: Iterable[str]) -> None:
    for index, example in enumerate(examples, start=1):
        print(f"{index}. {example}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a LangGraph research agent with Tavily web search and PDF download tools."
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Research or analysis question for the agent to answer.",
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
