# Poe Conversation Analysis Toolkit

This repository contains two parts:

1. `processing.py` — command-line pipeline that ingests raw conversation JSON, computes metrics, labels topics/intent with Poe or heuristic fallbacks, and writes results to the `outputs/` directory.
2. `dashboard/` — static web dashboard that visualises the processed metrics locally.

## 1. Generate the processed data

```
python3 processing.py \
  --input conversations.json \
  --output-dir outputs \
  --limit 60 \
  --cache-path outputs/topic_cache.json \
  --poe-api-key "$POE_API_KEY"
```

- Set the `POE_API_KEY` environment variable or pass `--poe-api-key` explicitly.
- Add `--disable-llm` to skip Poe and rely on keyword heuristics.

## 2. Launch the dashboard locally

The dashboard expects the JSON files produced above in `outputs/`.

```
python3 -m http.server 8000
```

Then open <http://localhost:8000/dashboard/> in your browser. The app fetches `outputs/aggregate_summary.json` and `outputs/per_conversation_metrics.json` at load time. Use the reload button to refresh after regenerating the data.

## Frontend overview

- `dashboard/index.html` — layout and DOM structure.
- `dashboard/styles.css` — styling with responsive summary cards, distributions, and table.
- `dashboard/app.js` — fetches processed JSON, renders visualisations, and adds table filtering.

Feel free to customise the dashboard styling or extend the visualisations with additional metrics.
