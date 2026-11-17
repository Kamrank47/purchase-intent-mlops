"""FastAPI inference service."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException

from .config import DEFAULT_META_PATH, DEFAULT_MODEL_PATH
from .schemas import Health, ModelInfo, Prediction, ShopperFeatures

THRESHOLD = 0.5


def _load_meta(model_file: Path) -> dict:
    meta_path = Path(os.getenv("MODEL_META_PATH", DEFAULT_META_PATH))
    if not meta_path.exists():
        sibling = model_file.parent / "model_meta.json"
        meta_path = sibling if sibling.exists() else meta_path
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Build the API. A missing model starts the app in a degraded state."""
    path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    state: dict = {"model": None, "error": None, "loaded_at": None}

    try:
        state["model"] = joblib.load(path)
        state["loaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    except Exception as exc:  # surfaced through /healthz instead of crashing
        state["error"] = str(exc)

    meta = _load_meta(path)

    app = FastAPI(
        title="Purchase Intent API",
        version="0.1.0",
        description="Predicts whether an e-commerce session will convert.",
    )

    @app.get("/healthz", response_model=Health, tags=["ops"])
    def healthz() -> Health:
        loaded = state["model"] is not None
        return Health(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            model_path=str(path),
        )

    @app.get("/model-info", response_model=ModelInfo, tags=["ops"])
    def model_info() -> ModelInfo:
        return ModelInfo(
            model_path=str(path),
            sklearn_version=sklearn.__version__,
            features=list(ShopperFeatures.model_fields),
            trained_at=meta.get("trained_at"),
            dataset_sha256=meta.get("dataset_sha256"),
        )

    @app.post("/predict", response_model=Prediction, tags=["inference"])
    def predict(payload: ShopperFeatures) -> Prediction:
        model = state["model"]
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Train it with `intent-train` or set MODEL_PATH.",
            )
        frame = pd.DataFrame([payload.as_record()])
        probability = float(model.predict_proba(frame)[:, 1][0])
        prediction = int(probability >= THRESHOLD)
        return Prediction(
            prediction=prediction,
            label="purchase" if prediction else "no_purchase",
            probability=round(probability, 6),
            threshold=THRESHOLD,
        )

    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    except Exception:  # optional monitoring extra
        pass

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "intent_mlops.serve:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
