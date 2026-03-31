from __future__ import annotations

import argparse
import operator
import os
from typing import Iterable

from dotenv import load_dotenv
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict


SYSTEM_PROMPT = """You are a careful research assistant.

Your job is to answer research and analysis questions with clear, structured, educational explanations.
When the topic is health related:
- Give general educational information.
- Call out uncertainty when appropriate.
- Avoid presenting the answer as medical diagnosis or personal medical advice.
- Suggest consulting a qualified clinician for individual medical concerns.
"""

HEALTH_RESEARCH_EXAMPLES = [
    "How does our body digest protein from the moment we eat it to the point where amino acids are used by cells?",
    "Explain how sleep quality influences muscle recovery and protein synthesis after exercise.",
    "Compare how soluble fiber and insoluble fiber affect digestion, satiety, and gut health.",
]


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def build_model() -> AzureChatOpenAI:
    load_dotenv()

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY is not set. Copy .env.example to .env and add your API key."
        )

    model_name = os.getenv("RESEARCH_AGENT_MODEL", "gpt-4o")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    return AzureChatOpenAI(
        azure_deployment=model_name, 
        azure_endpoint=endpoint, 
        api_key=api_key,
        api_version="2023-03-15-preview")


def build_agent():
    model = build_model()

    def research_node(state: ResearchState) -> ResearchState:
        response = model.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        )
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    graph = StateGraph(ResearchState)
    graph.add_node("researcher", research_node)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", END)
    return graph.compile()


def run_query(query: str) -> dict:
    agent = build_agent()
    return agent.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "llm_calls": 0,
        }
    )


def print_response(result: dict) -> None:
    final_message = result["messages"][-1]
    if isinstance(final_message, AIMessage):
        print(final_message.content)
        print(f"\nLLM calls: {result['llm_calls']}")
        return

    print(final_message)


def show_examples(examples: Iterable[str]) -> None:
    for index, example in enumerate(examples, start=1):
        print(f"{index}. {example}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a basic LangGraph research agent built with the Graph API."
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
