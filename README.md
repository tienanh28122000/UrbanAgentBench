# UrbanAgentBench

UrbanAgentBench is a benchmark for evaluating AI agents on urban understanding and decision-making tasks. It provides two domains that test an agent's ability to reason about real-world urban environments through satellite imagery and web-based map services.

## Domains

### Urban-Satellite
Evaluates agents on satellite imagery analysis tasks for urban infrastructure assessment. The agent must use vision-language capabilities to:
- Analyze current and historical satellite images of urban sites
- Detect infrastructure changes over time (construction, demolition, land use shifts)
- Classify land use types and count objects in aerial views
- Answer questions about site conditions based on visual evidence

### Urban-Map-Web
Evaluates agents on web-based map and local services tasks. The agent must:
- Search for places, points of interest, and routes in urban environments
- Retrieve detailed information about venues (hours, ratings, reviews)
- Manage bookings and reservations for restaurants and services
- Plan multi-stop itineraries and navigate between locations

## Installation

**Requirements**: Python 3.10+

```bash
git clone <repo-url>
cd UrbanAgentBench
pip install -e .
```

Copy the environment template and add your API keys:

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY, OPENROUTER_API_KEY, etc.
```

## Quick Start

Run the Urban-Satellite benchmark:

```bash
bash scripts/run_urban_satellite.sh
```

Run the Urban-Map-Web benchmark:

```bash
bash scripts/run_urban_map_web.sh
```

Or run directly via CLI:

```bash
uab run --domain urban_satellite --agent-llm openrouter/openai/gpt-4.1-mini --user-llm openrouter/openai/gpt-4.1-mini --num-tasks 10
uab run --domain urban_map_web --agent-llm openrouter/openai/gpt-4.1-mini --user-llm openrouter/openai/gpt-4.1-mini --num-tasks 10
```

## CLI Reference

```
uab run --help
```

Key options:
- `--domain`: `urban_satellite` or `urban_map_web`
- `--agent-llm`: Model for the agent (e.g., `openrouter/openai/gpt-4.1-mini`)
- `--user-llm`: Model for the user simulator
- `--num-tasks`: Number of tasks to evaluate
- `--num-trials`: Number of trials per task
- `--agent-llm-args`: JSON string of extra args (e.g., `'{"max_tokens": 2048}'`)
- `--save-to`: Output file name (saved under `data/simulations/`)

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `UAB_DATA_DIR` | (Optional) custom path to data directory |

## Repository Structure

```
UrbanAgentBench/
├── src/urban_agent_bench/   # Core Python package
│   ├── domains/
│   │   ├── urban_satellite/ # Satellite imagery domain
│   │   └── urban_map_web/   # Map/web services domain
│   ├── agent/               # LLM agent implementations
│   ├── user/                # User simulator
│   ├── evaluator/           # Evaluation framework
│   └── ...
├── data/
│   └── domains/
│       ├── urban_satellite/ # Benchmark data, tasks, satellite images
│       └── urban_map_web/   # Benchmark data, tasks, place database
└── scripts/
    ├── run_urban_satellite.sh
    └── run_urban_map_web.sh
```
