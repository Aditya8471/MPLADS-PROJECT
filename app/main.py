from __future__ import annotations

import csv
import io
import json
import math
import statistics
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data.json"

app = FastAPI(
    title="MPLADS Sentinel API",
    version="1.0.0",
    description="Explainable anomaly and data-quality backend for the MPLADS Sentinel prototype.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin before production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WorkRecord(BaseModel):
    work_id: str
    house: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    mp_name: Optional[str] = None
    recommended_amount: Optional[float] = Field(default=None, ge=0)
    sanctioned_amount: Optional[float] = Field(default=None, ge=0)
    expenditure: Optional[float] = Field(default=None, ge=0)
    completion_pct: Optional[float] = Field(default=None, ge=0, le=100)
    sanction_date: Optional[str] = None
    completion_date: Optional[str] = None
    implementing_agency: Optional[str] = None

def load_records() -> list[dict[str, Any]]:
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)

RECORDS = load_records()

def robust_stats(values: list[float]) -> tuple[float, float, float]:
    """Return median, MAD and robust sigma (1.4826 * MAD)."""
    if not values:
        return 0.0, 0.0, 0.0
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    sigma = 1.4826 * mad
    return med, mad, sigma

def enrich(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [r["allocated"] for r in records if isinstance(r.get("allocated"), (int, float))
             and math.isfinite(r["allocated"]) and r["allocated"] > 0]
    median, mad, robust_sigma = robust_stats(valid)

    sr_counts: dict[str, int] = {}
    name_counts: dict[str, int] = {}
    for r in records:
        sr_key = f'{r["house"]}|{r["sr_no"]}'
        nm_key = f'{r["house"]}|{r["mp"].strip().lower()}|{r["state"].strip().lower()}'
        sr_counts[sr_key] = sr_counts.get(sr_key, 0) + 1
        name_counts[nm_key] = name_counts.get(nm_key, 0) + 1

    output = []
    for r0 in records:
        r = dict(r0)
        reasons: list[str] = []
        score = 0.0
        critical = False

        amount = r.get("allocated")
        if amount is None:
            critical = True
            score += 60
            reasons.append("Allocated amount is blank in the published dataset.")
        elif not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= 0:
            critical = True
            score += 60
            reasons.append("Allocated amount is not a valid positive number.")
        else:
            # Modified/robust z-score: resistant to unusually large allocations.
            if robust_sigma > 0:
                mz = 0.6745 * (amount - median) / mad if mad > 0 else 0.0
                if abs(mz) >= 3.5:
                    score += min(35.0, 10.0 + (abs(mz) - 3.5) * 5.0)
                    direction = "above" if amount > median else "below"
                    reasons.append(
                        f"Robust statistical outlier: allocation is {direction} the normal range "
                        f"(modified z-score {mz:.2f})."
                    )

        sr_key = f'{r["house"]}|{r["sr_no"]}'
        nm_key = f'{r["house"]}|{r["mp"].strip().lower()}|{r["state"].strip().lower()}'
        if sr_counts[sr_key] > 1:
            score += 25
            reasons.append(f'Serial number {r["sr_no"]} occurs {sr_counts[sr_key]} times within {r["house"]}.')
        if name_counts[nm_key] > 1:
            score += 25
            reasons.append("MP name + state combination occurs more than once within the same House.")

        score = min(100.0, round(score, 1))
        if critical:
            level = "critical"
        elif score >= 60:
            level = "high"
        elif score >= 30:
            level = "medium"
        else:
            level = "low"

        r.update({
            "issue_level": level,
            "flagged": bool(reasons),
            "risk_score": score,
            "reasons": reasons,
            "anomaly": {
                "method": "ROBUST_MAD",
                "median": round(median, 2),
                "mad": round(mad, 2),
                "robust_sigma": round(robust_sigma, 2),
            },
        })
        output.append(r)
    return output

def filtered(
    house: str = "both",
    state: str = "all",
    issue: str = "all",
    q: str = "",
) -> list[dict[str, Any]]:
    rows = enrich(RECORDS)
    q = q.strip().lower()
    result = []
    for r in rows:
        if house != "both" and r["house"] != house:
            continue
        if state != "all" and r["state"] != state:
            continue
        if issue == "flagged" and not r["flagged"]:
            continue
        if q:
            hay = " ".join(filter(None, [
                r["mp"], r["state"], r.get("constituency") or "",
                r.get("elected_nominated") or ""
            ])).lower()
            if q not in hay:
                continue
        result.append(r)
    return result

@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(BASE / "static.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "mplads-sentinel-api", "records": len(RECORDS)}

@app.get("/api/meta")
def meta():
    rows = enrich(RECORDS)
    return {
        "houses": {"LS": sum(r["house"] == "LS" for r in rows), "RS": sum(r["house"] == "RS" for r in rows)},
        "states": sorted({r["state"] for r in rows}),
        "records": len(rows),
        "method": "ROBUST_MAD + EXPLAINABLE_RULES",
        "source": "Published eSAKSHI/MPLADS allocation dataset supplied with the prototype",
    }

@app.get("/api/records")
def get_records(
    house: str = "both",
    state: str = "all",
    issue: str = "all",
    q: str = "",
    limit: int = 1000,
    offset: int = 0,
):
    rows = filtered(house, state, issue, q)
    return {
        "items": rows[offset:offset + min(limit, 5000)],
        "total": len(rows),
        "offset": offset,
        "limit": limit,
    }

@app.get("/api/records/{record_id}")
def get_record(record_id: str):
    for r in enrich(RECORDS):
        if r["id"] == record_id:
            return r
    raise HTTPException(status_code=404, detail="Record not found")

@app.get("/api/dashboard")
def dashboard(house: str = "both", state: str = "all", issue: str = "all", q: str = ""):
    rows = filtered(house, state, issue, q)
    allocated = [r["allocated"] for r in rows if isinstance(r.get("allocated"), (int, float))]
    flagged = [r for r in rows if r["flagged"]]
    by_house = {}
    by_state = {}
    for r in rows:
        by_house.setdefault(r["house"], {"records": 0, "allocated": 0})
        by_house[r["house"]]["records"] += 1
        by_house[r["house"]]["allocated"] += r["allocated"] or 0
        by_state.setdefault(r["state"], {"records": 0, "flagged": 0, "allocated": 0})
        by_state[r["state"]]["records"] += 1
        by_state[r["state"]]["flagged"] += int(r["flagged"])
        by_state[r["state"]]["allocated"] += r["allocated"] or 0

    return {
        "scope": {"house": house, "state": state, "issue": issue, "q": q},
        "stats": {
            "records": len(rows),
            "flagged": len(flagged),
            "critical": sum(r["issue_level"] == "critical" for r in rows),
            "high": sum(r["issue_level"] == "high" for r in rows),
            "medium": sum(r["issue_level"] == "medium" for r in rows),
            "clean": sum(not r["flagged"] for r in rows),
            "allocated_total": sum(allocated),
            "allocated_average": statistics.mean(allocated) if allocated else 0,
            "states": len({r["state"] for r in rows}),
        },
        "risk": {
            "mean_score": statistics.mean([r["risk_score"] for r in rows]) if rows else 0,
            "high_priority": sum(r["risk_score"] >= 60 for r in rows),
            "statistical_outliers": sum(any("statistical outlier" in x.lower() for x in r["reasons"]) for r in rows),
        },
        "by_house": by_house,
        "by_state": by_state,
        "priority_alerts": sorted(flagged, key=lambda r: (-r["risk_score"], r["mp"]))[:100],
    }

@app.get("/api/alerts")
def alerts(limit: int = 100, min_score: float = 0):
    rows = [r for r in enrich(RECORDS) if r["flagged"] and r["risk_score"] >= min_score]
    return {"items": sorted(rows, key=lambda r: -r["risk_score"])[:limit]}

@app.get("/api/states")
def states(house: str = "both"):
    rows = enrich(RECORDS)
    return {"items": sorted({r["state"] for r in rows if house == "both" or r["house"] == house})}

@app.get("/api/export.csv")
def export_csv(house: str = "both", state: str = "all", issue: str = "all", q: str = ""):
    rows = filtered(house, state, issue, q)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id","sr_no","house","mp","state","constituency","elected_nominated","allocated",
                     "issue_level","flagged","risk_score","reasons"])
    for r in rows:
        writer.writerow([
            r["id"], r["sr_no"], r["house"], r["mp"], r["state"], r.get("constituency"),
            r.get("elected_nominated"), r.get("allocated"), r["issue_level"], r["flagged"],
            r["risk_score"], " | ".join(r["reasons"])
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="mplads-sentinel-export.csv"'},
    )

@app.post("/api/work-records/score")
def score_work(record: WorkRecord):
    """
    Optional implementation-layer scoring endpoint.
    It does not invent work data: it scores work-level data supplied by an operator.
    """
    flags: list[str] = []
    score = 0.0
    if record.sanctioned_amount is not None and record.recommended_amount not in (None, 0):
        variance = (record.sanctioned_amount - record.recommended_amount) / record.recommended_amount * 100
        if abs(variance) >= 20:
            score += min(30, abs(variance) / 2)
            flags.append(f"Sanction/recommendation variance is {variance:.1f}%.")
    if record.expenditure is not None and record.sanctioned_amount not in (None, 0):
        util = record.expenditure / record.sanctioned_amount * 100
        if util < 25:
            score += 20
            flags.append(f"Low expenditure utilization: {util:.1f}%.")
        elif util > 100:
            score += 40
            flags.append(f"Expenditure exceeds sanctioned amount: {util:.1f}%.")
    if record.completion_pct is not None and record.completion_pct < 50:
        score += 15
        flags.append("Work completion is below 50%.")
    if record.implementing_agency:
        flags.append("Agency supplied; cross-record concentration analysis can be run when multiple work records are uploaded.")

    return {
        "work_id": record.work_id,
        "risk_score": min(100, round(score, 1)),
        "flags": flags,
        "status": "flagged" if score >= 30 else "review",
        "note": "Heuristic implementation-layer screening, not proof of fraud.",
    }

@app.post("/api/work-records/batch-score")
def batch_score(records: list[WorkRecord]):
    return {"items": [score_work(r) for r in records], "count": len(records)}

@app.post("/api/work-records/upload")
async def upload_work_csv(file: UploadFile = File(...)):
    """
    Accepts an optional implementation CSV. Expected headers:
    work_id,house,state,district,mp_name,recommended_amount,sanctioned_amount,
    expenditure,completion_pct,sanction_date,completion_date,implementing_agency
    """
    raw = await file.read()
    try:
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc
    if not rows:
        raise HTTPException(status_code=400, detail="CSV contains no rows.")

    parsed = []
    for row in rows:
        def num(key):
            value = row.get(key)
            if value in (None, ""):
                return None
            return float(value)

        parsed.append(WorkRecord(
            work_id=row.get("work_id") or "",
            house=row.get("house") or None,
            state=row.get("state") or None,
            district=row.get("district") or None,
            mp_name=row.get("mp_name") or None,
            recommended_amount=num("recommended_amount"),
            sanctioned_amount=num("sanctioned_amount"),
            expenditure=num("expenditure"),
            completion_pct=num("completion_pct"),
            sanction_date=row.get("sanction_date") or None,
            completion_date=row.get("completion_date") or None,
            implementing_agency=row.get("implementing_agency") or None,
        ))
    return {"filename": file.filename, "count": len(parsed), "items": [score_work(r) for r in parsed]}

@app.post("/api/reload")
def reload_data():
    global RECORDS
    RECORDS = load_records()
    return {"status": "reloaded", "records": len(RECORDS)}
