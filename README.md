# Grant Scoring: Rules + ML

Two-mode grant scoring: transparent business rules + calibrated logistic regression. Includes scenario generator, CSV evaluation with bucketed metrics, FastAPI API, and Streamlit UI.

## Stack
Python 3.10+, FastAPI, scikit-learn, pandas, Streamlit, pytest.

## Rules
- **Hard declines:** `parents_status=="high"`; `is_immigrant and residency_years_ie>=3.0`.
- **Base:** `parents_status=="low"` → approve; `parents_status=="middle"` → approve iff `distance_km>30`.
- **Amount:** `<=30km → 2560`, else `5697`.

## Repo
```
app/{api,domain,ml,models,services,ui}
data/{raw,processed}
docs/img
tests
```

## Install
```bash
python -m venv .venv && . .venv/Scripts/Activate.ps1   # on Windows
pip install -r requirements.txt
```

## TL;DR
```bash
# generate labeled scenarios (with approved + bucket)
python -m app.ml.generate_scenarios

# train model → app/models/model.pkl
python -m app.ml.train

# evaluate on CSV → data/processed/{metrics_from_csv.json,preds_from_csv.csv}
python -m app.ml.eval_from_csv
```

> Note: `generate_data` produces *unlabeled* CSV and can overwrite `data/raw/generated_applicants.csv`. Use scenarios for labeled evaluation.

## API
Run:
```bash
uvicorn app.api.main:app --reload --port 8000
```
OpenAPI: `http://localhost:8000/docs`

Score example:
```bash
curl -X POST http://localhost:8000/score   -H "Content-Type: application/json"   -d '{
    "distance_km": 20,
    "residency_years_ie": 2.0,
    "is_immigrant": true,
    "parents_status": "low"
  }'
```
Response:
```json
{
  "approved": true,
  "decision": "approve",
  "amount": 2560,
  "prob": 0.68,
  "reasons": ["low income", "ml ≥ thr"]
}
```

## UI
```bash
streamlit run app/ui/dashboard.py
```
Shows mode (Rules/ML, threshold), decision, probability bar, amount, and reasons. Add screenshots to `docs/img/` and reference:
```markdown

![Approve](img/ui_approve.png?v=1)
![Decline override](img/ui_decline_override.png?v=1)

```

## Tests
```bash
pytest -q
```

## License
MIT.
