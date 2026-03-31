# LangGraph Research Agent

This project is a LangGraph Graph API research agent in Python. It now includes tool calling for web research and PDF collection.

## What it does

- Uses LangGraph `StateGraph` with explicit nodes and edges.
- Uses `tavily_web_search` for web search tasks that need up-to-date sources.
- Uses `download_pdfs` to download PDF URLs into a local `papers` folder.
- Supports health-focused research prompts, including protein digestion workflows.

## Project files

- `src/research_agent.py`: Graph API agent, Tavily search tool, and PDF download tool.
- `.env.example`: Azure OpenAI + Tavily environment variable template.
- `requirements.txt`: Runtime dependencies.

## Setup

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install or refresh dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create local env file:

```powershell
Copy-Item .env.example .env
```

Set values in `.env`:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION` (default in template is `2024-02-01`)
- `RESEARCH_AGENT_MODEL` (Azure deployment name, default `gpt-4o`)
- `TAVILY_API_KEY`

## Run the agent

Show built-in examples:

```powershell
.\.venv\Scripts\python.exe .\src\research_agent.py --examples
```

Run a basic prompt:

```powershell
.\.venv\Scripts\python.exe .\src\research_agent.py "How does our body digest protein?"
```

Run a web-research prompt that should trigger both tools:

```powershell
.\.venv\Scripts\python.exe .\src\research_agent.py "Find recent credible research papers and PDF links about protein digestion, download the PDFs, then summarize the findings with source URLs."
```

PDFs are stored in the local `papers` folder.

## Graph flow

1. `llm_call`: model decides whether to respond directly or call a tool.
2. `tool_node`: executes requested tool calls (`tavily_web_search`, `download_pdfs`).
3. Conditional edge loops back to `llm_call` until no more tool calls remain.
