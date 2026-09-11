# MPLADS Sentinel — Full Backend

FastAPI backend for the supplied MPLADS Sentinel frontend.

## Run

```bash
cd app
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r ../requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/ for the dashboard.
API docs: http://localhost:8000/docs

## API

- `GET /api/health`
- `GET /api/meta`
- `GET /api/records`
- `GET /api/records/{id}`
- `GET /api/dashboard`
- `GET /api/alerts`
- `GET /api/states`
- `GET /api/export.csv`
- `POST /api/work-records/score`
- `POST /api/work-records/batch-score`
- `POST /api/work-records/upload`
- `POST /api/reload`

## Intelligence

The allocation layer uses a robust MAD/modified-z-score method plus explainable data-quality rules. The optional implementation endpoint scores supplied work-level records for sanction variance, expenditure utilization and completion lag.

The implementation endpoint is deliberately input-driven: it does not manufacture official work-level data.
