# UniScientist

<p align="center">
  <a href="https://unipat.ai/blog/UniScientist"><b>Blog</b></a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/UnipatAI"><b>Models</b></a> &nbsp;|&nbsp;
  <a><b>Paper (Coming Soon)</b></a>
</p>

UniScientist is designed to advance universal scientific research intelligence through a unified paradigm. Leveraging an evolving polymathic synthesis, it generates research-grade data that enables structured, rubric-based supervision.

## Project Structure

```
UniScientist/
├── local_deploy.sh                 # Step 1: Deploy local LLM via vLLM
├── inference_local_qwen.sh         # Step 2: Run agentic inference (repeat for multiple rollouts)
├── inference_local_qwen.py         # Agentic inference engine
├── inference_local_aggregate.py    # Step 3: Aggregate multiple rollouts into a final report
├── tools/
│   ├── tool_search.py              # Google web search (via Serper API)
│   ├── tool_scholar.py             # Google Scholar search (via Serper API)
│   ├── tool_visit.py               # Webpage reader (via Jina Reader API) with LLM summarization
│   └── tool_code.py                # Python code interpreter
├── data/                           # Place your input data here (see Data Format below)
├── requirements.txt
└── .gitignore
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain API Keys

The following API keys are required for the tool suite:

| Key | Service | Purpose |
|-----|---------|---------|
| `SERPER_KEY_ID` | [Serper](https://serper.dev/) | Google web search & Google Scholar |
| `JINA_API_KEYS` | [Jina Reader](https://jina.ai/reader/) | Webpage content reading |
| `OPENROUTER_API_KEY` | [OpenRouter](https://openrouter.ai/) | LLM-based webpage summarization |

### 3. Prepare Data

Place your input data in the `data/` directory as `.jsonl` files. Each line should be a JSON object with the following fields:

```json
{"problem": "Your research question here", "answer": "Ground truth answer / rubrics (optional)"}
```

## Usage

The workflow consists of three steps:

### Step 1: Deploy Local LLM

Edit `local_deploy.sh` to set `MODEL_PATH` to your model weights, then:

```bash
bash local_deploy.sh
```

This starts a vLLM OpenAI-compatible server on port 8000. Wait until the server is ready before proceeding.

### Step 2: Run Agentic Inference

Edit `inference_local_qwen.sh` to fill in your API keys and configuration, then run it **multiple times** to collect diverse rollouts:

```bash
# Run N times to collect N rollouts
bash inference_local_qwen.sh
bash inference_local_qwen.sh
bash inference_local_qwen.sh
```

Each run produces (or appends to) a `.jsonl` output file named `<STORED_MODEL_NAME>_<BENCHMARK>.jsonl`.

### Step 3: Aggregate Results

Merge multiple rollout results into a single comprehensive report:

```bash
python inference_local_aggregate.py \
    --model-name "qwen3-235b" \
    --data-paths rollout_1.jsonl rollout_2.jsonl rollout_3.jsonl \
    --benchmark research \
    --rollout-num 1 \
    --llm-max-concurrency 32
```

#### Aggregation Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-name` | Yes | - | Model identifier for naming the output file |
| `--data-paths` | Yes | - | One or more rollout `.jsonl` files to aggregate |
| `--benchmark` | No | `research` | Benchmark name for naming the output file |
| `--rollout-num` | No | `1` | Number of aggregation passes per question cluster |
| `--local-base-url` | No | `http://localhost:8000/v1` | vLLM server endpoint |
| `--output-path` | No | auto-generated | Custom output file path |
| `--llm-max-concurrency` | No | `32` | Max concurrent LLM requests |

## Citation

If you find UniScientist useful in your research, please cite:

```bibtex
@misc{unipat2026uniscientist,
  title   = {UniScientist: Advancing Universal Scientific Research Intelligence},
  author  = {UniPat AI Team},
  year    = {2026},
  url     = {https://unipat.ai/blog/UniScientist}
}
```

## Contact

We are continuously expanding the Universal Scientific Research Dataset to cover additional disciplines and research paradigms. We welcome collaborations with research teams interested in advancing scientific research intelligence. Reach out at contact@unipat.ai.
