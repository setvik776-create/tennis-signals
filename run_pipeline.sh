#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/root/tennis_signals"
PYTHON="$BASE_DIR/.venv/bin/python"
TARGET="${1:-tomorrow}"
MATCHES="$BASE_DIR/data/target_matches.csv"
PREDICTIONS="$BASE_DIR/data/predictions.csv"
MODEL="$BASE_DIR/data/tennis_model.joblib"
TRACKER="$BASE_DIR/data/prediction_tracker.csv"

if [[ "$TARGET" != "today" && "$TARGET" != "tomorrow" ]]; then
  echo "Usage: $0 today|tomorrow" >&2
  exit 2
fi

cd "$BASE_DIR"
mkdir -p "$BASE_DIR/data" "$BASE_DIR/logs"

"$PYTHON" "$BASE_DIR/scripts/scraper.py" --target "$TARGET" --output "$MATCHES"

if ! "$PYTHON" - "$MATCHES" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
try:
    df = pd.read_csv(path)
except Exception:
    sys.exit(1)
sys.exit(0 if len(df) > 0 else 1)
PY
then
  echo "No target matches found in $MATCHES; pipeline finished without predictions."
  exit 0
fi

"$PYTHON" "$BASE_DIR/scripts/predictor.py" \
  --history "$BASE_DIR/data/tennis_all_matches_2024_to_now.csv" \
  --matches "$MATCHES" \
  --output "$PREDICTIONS" \
  --model-out "$MODEL"

"$PYTHON" - "$PREDICTIONS" "$TRACKER" "$TARGET" <<'PY'
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd

predictions_path = Path(sys.argv[1])
tracker_path = Path(sys.argv[2])
target = sys.argv[3]

predictions = pd.read_csv(predictions_path)
predictions.insert(0, "target", target)
predictions.insert(0, "run_at", datetime.utcnow().isoformat(timespec="seconds") + "Z")
for column in ["actual_winner", "is_correct"]:
    if column not in predictions.columns:
        predictions[column] = ""

tracker_path.parent.mkdir(parents=True, exist_ok=True)
if tracker_path.exists():
    tracker = pd.read_csv(tracker_path)
    combined = pd.concat([tracker, predictions], ignore_index=True)
else:
    combined = predictions

dedupe_columns = ["target", "match_date", "player1", "player2", "predicted_winner"]
combined = combined.drop_duplicates(subset=dedupe_columns, keep="last")
combined.to_csv(tracker_path, index=False)
print(f"Tracker updated: {tracker_path} rows={len(combined)}")
PY

"$PYTHON" "$BASE_DIR/scripts/broadcaster.py" --predictions "$PREDICTIONS"
