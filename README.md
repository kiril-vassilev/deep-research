# LangGraph Research Agent

This project is a LangGraph Graph API research agent in Python. It now includes tool calling for web research and PDF collection.

## What it does

- Uses LangGraph `StateGraph` with explicit nodes and edges.
- Uses `tavily_web_search` for web search tasks that need up-to-date sources.
- Uses `download_pdfs` to download PDF URLs into a local `papers` folder.
- Operates autonomously toward the goal without asking follow-up questions.
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
- `MAX_AUTONOMOUS_STEPS` (optional, default `10`)

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

## Step 1 input quality

Provide complete instructions in your first prompt so the agent can execute independently.

Recommended structure:

- `Research goal`: what question to answer.
- `Scope`: date range, geography, population, or domain constraints.
- `Source policy`: what source quality to prioritize (peer-reviewed, government, standards bodies).
- `Deliverable`: expected output format (summary, comparison table, recommendations, risks).

Example first input:

```text
Research goal: Analyze how protein digestion efficiency varies by age group in healthy adults.
Scope: Prioritize studies from 2018 onward, human studies only, include at least one systematic review.
Source policy: Prefer peer-reviewed papers and government or university sources.
Deliverable: Provide a structured report with key findings, conflicting evidence, a short limitations section, and source URLs. Download available PDFs to the papers folder.
```

## Graph flow

1. `llm_call`: model plans and decides whether to use tools.
2. `tool_node`: executes requested tool calls (`tavily_web_search`, `download_pdfs`).
3. `autonomy_nudge`: if goal is still in progress, pushes another autonomous iteration.
4. Loop continues until `GOAL_STATUS: COMPLETE` or `MAX_AUTONOMOUS_STEPS` is reached.
