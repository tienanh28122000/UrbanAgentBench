# UrbanAgentBench

UrbanAgentBench is a benchmark for evaluating AI agents on urban reasoning and decision-making tasks.

It includes two domains:
- `urban_map_web`: map, place, route, and booking workflows
- `urban_satellite`: satellite-image reasoning and assessment

## Requirements

- Python 3.10+
- API key(s) for the LLM providers you use

## Installation

```bash
git clone <repo-url>
cd UrbanAgentBench
pip install -e .
cp .env.example .env
# Edit .env and set OPENAI_API_KEY / OPENROUTER_API_KEY / ...
```

## Run Benchmarks

Use helper scripts:

```bash
bash scripts/run_urban_satellite.sh
bash scripts/run_urban_map_web.sh
```

Or run from CLI:

```bash
uab run --domain urban_satellite --agent-llm openrouter/openai/gpt-5.4 --user-llm openrouter/openai/gpt-5.4 --num-tasks 10
uab run --domain urban_map_web --agent-llm openrouter/openai/gpt-5.4 --user-llm openrouter/openai/gpt-5.4 --num-tasks 10
```

## Run Visualization (One Command)

```bash
bash scripts/run_visualization.sh
```

Default URLs:
- Landing: `http://127.0.0.1:8000/visualization/index.html`
- Urban-Map-Web: `http://127.0.0.1:8000/visualization/urban-map-web.html`
- Urban-Satellite: `http://127.0.0.1:8000/visualization/urban-satellite.html`

Custom port:

```bash
bash scripts/run_visualization.sh 8088
```

## Update Leaderboard Data

From `UrbanAgentBench/visualization`:

```bash
python3 metrics/calculate_leaderboard.py --action_threshold 0 --nl_threshold 0
```

Notes:
- If `--domain/--domains` is omitted, both domains are processed.
- Accepted forms include:
    - `--domain urban_map_web urban_satellite`
    - `--domains urban_map_web,urban_satellite`
    - `--domains [urban_map_web,urban_satellite]`

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `UAB_DATA_DIR` | Optional custom path to benchmark data |

## Repository Structure

```text
UrbanAgentBench/
├── src/urban_agent_bench/         # Core benchmark package
├── data/domains/
│   ├── urban_satellite/           # Tasks, policies, satellite images
│   └── urban_map_web/             # Tasks and place database
├── task_generator/                # Crawl/build static DB and task resources for domains
├── visualization/                 # Static UI for trajectories + leaderboard
├── scripts/
│   ├── run_urban_satellite.sh
│   ├── run_urban_map_web.sh
│   └── run_visualization.sh
└── README.md
```

## Additional Docs

- Visualization-specific docs are kept in `visualization/README.md`.
