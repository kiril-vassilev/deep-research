# LangGraph Research Agent

This project is a LangGraph Graph API research agent in Python with three integrated tools for comprehensive research.

## What it does

- Uses LangGraph `StateGraph` with explicit nodes and edges.
- Uses `tavily_web_search` for web search tasks that need up-to-date sources.
- Uses `download_pdfs` to download PDF URLs into a local `papers` folder.
- Uses `parse_pdfs_for_sections` to extract structured sections (abstract, methods, findings, discussion) from downloaded PDFs.
- Operates autonomously toward the goal without asking follow-up questions.
- Synthesizes final responses using extracted PDF content for stronger evidence.
- Supports health-focused research prompts, including protein digestion workflows.

## Project files

- `src/research_agent.py`: Graph API agent with three integrated research tools.
- `.env.example`: Azure OpenAI + Tavily environment variable template.
- `requirements.txt`: Runtime dependencies including pypdf for PDF parsing.

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

Run a full research prompt with all three tools:

```powershell
.\.venv\Scripts\python.exe .\src\research_agent.py "Research goal: Analyze how protein digestion efficiency varies by age group in healthy adults. Scope: Prioritize studies from 2018 onward, human studies only, include at least one systematic review. Source policy: Prefer peer-reviewed papers and government or university sources. Deliverable: Structured report with key findings, conflicting evidence, limitations, and source URLs. Download available PDFs and extract their key sections."
```

PDFs are stored in the local `papers` folder. Extracted sections are synthesized into the final response.

## Tools overview

1. **tavily_web_search**: Searches for credible research sources and PDFs with enhanced query enrichment.
2. **download_pdfs**: Downloads PDF files from URLs with validation and safe naming.
3. **parse_pdfs_for_sections**: Extracts abstract, methods, findings, discussion, and conclusion sections from PDFs using regex-based section detection.

## Step 1 input quality

Provide complete instructions in your first prompt so the agent can execute independently.

Recommended structure:

- `Research goal`: what question to answer.
- `Scope`: date range, geography, population, or domain constraints.
- `Source policy`: what source quality to prioritize (peer-reviewed, government, standards bodies).
- `Deliverable`: expected output format (summary, comparison table, recommendations, risks). Mention PDF extraction if desired.

Example first input:

```text
Research goal: Analyze how protein digestion efficiency varies by age group in healthy adults.
Scope: Prioritize studies from 2018 onward, human studies only, include at least one systematic review.
Source policy: Prefer peer-reviewed papers and government or university sources.
Deliverable: Provide a structured report with key findings, conflicting evidence, a short limitations section, and source URLs. Download available PDFs and extract key sections to strengthen the synthesis.
```

## Graph flow

1. `llm_call`: model plans and decides whether to use tools.
2. `tool_node`: executes requested tool calls (`tavily_web_search`, `download_pdfs`, `parse_pdfs_for_sections`).
3. `autonomy_nudge`: if goal is still in progress, pushes another autonomous iteration.
4. Loop continues until `GOAL_STATUS: COMPLETE` or `MAX_AUTONOMOUS_STEPS` is reached.
