# LangGraph Research Agent

This project starts with a basic research and analysis agent built with the LangGraph Graph API in Python. It does not use tools yet, so the graph is intentionally simple: a single model node receives the conversation state and returns a response.

## What it does

- Uses LangGraph's `StateGraph` API instead of the functional API.
- Accepts a research-style query and returns a structured answer.
- Includes health-focused example prompts, including protein digestion.
- Keeps the design easy to extend with tools in a later step.

## Project files

- `src/research_agent.py`: Graph API agent implementation and CLI entry point.
- `.env.example`: Environment variables for the model configuration.
- `requirements.txt`: Minimal runtime dependencies.

## Setup

The virtual environment for this project was created at `.venv` before dependencies were installed.

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Create your local environment file:

```powershell
Copy-Item .env.example .env
```

Then add your OpenAI API key to `.env`.

## Run the agent

List the built-in example prompts:

```powershell
.\.venv\Scripts\python.exe .\src\research_agent.py --examples
```

Run the protein digestion example:

```powershell
.\.venv\Scripts\python.exe .\src\research_agent.py "How does our body digest protein from the moment we eat it to the point where amino acids are used by cells?"
```

Run a different research question:

```powershell
.\.venv\Scripts\python.exe .\src\research_agent.py "Compare how soluble fiber and insoluble fiber affect digestion, satiety, and gut health."
```

## How the graph is structured

The agent follows the Graph API pattern from the LangGraph quickstart:

1. Define a typed state.
2. Define a node that calls the model.
3. Add `START -> researcher -> END` edges.
4. Compile the graph and invoke it with a message list.

This gives you a clean base for the next step, where we can add tools and conditional routing.