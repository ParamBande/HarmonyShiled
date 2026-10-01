"""
HarmonyShield FastAPI Backend
Wraps the existing frozen inference pipeline. No models are retrained.
"""

import os
import sys
import time
import uuid
import tempfile
import shutil
import logging
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import joblib
import torch

from src.audio.preprocessor import AudioPreprocessor
from src.features.extractor import FeatureExtractor
from src.features.melspectrogram import MelSpectrogramExtractor
from src.models.resnet18_model import AudioResNet18
from src.pretrained import ASTDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("harmonyshield")

# ─── Model store (loaded once at startup) ────────────────────────────────────
_models: dict = {}

RESNET_CHECKPOINTS = [
    ("audioldm2",        "LOGO_audioldm2_model.pt"),
    ("stable_audio_open","LOGO_stable_audio_open_model.pt"),
    ("musicldm",         "LOGO_musicldm_model.pt"),
    ("mustango",         "LOGO_mustango_model.pt"),
    ("MusicGen_medium",  "LOGO_MusicGen_medium_model.pt"),
]

CKPT_DIR = os.path.join(PROJECT_ROOT, "experiments", "resnet18_generalization", "checkpoints")
RF_PATH  = os.path.join(PROJECT_ROOT, "data", "pilot_dataset", "best_rf_model.joblib")

# Classical RF feature importances (frozen, loaded once)
RF_FEATURE_NAMES: list = []


def load_all_models():
    global RF_FEATURE_NAMES
    logger.info("Loading models (this may take a moment)…")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")
    _models["device"] = device

    # Preprocessing
    _models["preprocessor"] = AudioPreprocessor(segment_duration=10.0, target_sr=16000)

    # Classical
    logger.info("Loading Classical Random Forest…")
    rf = joblib.load(RF_PATH)
    _models["rf"] = rf
    _models["classical_extractor"] = FeatureExtractor(sr=16000)
    # Always generate feature names from extractor to ensure chart is populated
    dummy = np.zeros(16000 * 10)
    _, names = FeatureExtractor(sr=16000).extract_features(dummy, feature_set="B")
    RF_FEATURE_NAMES = names
    _models["rf_importances"] = None
    # Handle both plain RF and sklearn Pipeline
    if hasattr(rf, "feature_importances_"):
        _models["rf_importances"] = rf.feature_importances_
    elif hasattr(rf, "named_steps") and "classifier" in rf.named_steps:
        clf = rf.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            _models["rf_importances"] = clf.feature_importances_

    # ResNet ensemble
    logger.info("Loading ResNet18 ensemble (5 checkpoints)…")
    _models["resnet_extractor"] = MelSpectrogramExtractor(sr=16000)
    resnet_models = {}
    for name, ckpt in RESNET_CHECKPOINTS:
        path = os.path.join(CKPT_DIR, ckpt)
        if not os.path.exists(path):
            logger.warning(f"Checkpoint missing: {path}")
            continue
        m = AudioResNet18(pretrained=False).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        resnet_models[name] = m
    _models["resnet_models"] = resnet_models

    # Plan B  (AST) – slow to load, do it last
    logger.info("Loading Plan B AST detector…")
    _models["ast"] = ASTDetector()

    logger.info("All models loaded ✓")


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield
    # cleanup if needed
    _models.clear()


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="HarmonyShield API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic models ─────────────────────────────────────────────────────────
class FileInfo(BaseModel):
    name: str
    size_bytes: int
    duration: float
    sample_rate: int
    num_segments: int

class ModelResult(BaseModel):
    prediction: str
    ai_probability: float
    human_probability: float
    error: Optional[str] = None

class ResNetResult(ModelResult):
    ensemble_size: int
    checkpoint_results: list  # [{"name": ..., "ai_probability": ...}]

class ConsensusResult(BaseModel):
    status: str   # "STRONG_CONSENSUS_AI" | "STRONG_CONSENSUS_HUMAN" | "MIXED" | "INCONCLUSIVE"
    message: str
    num_ai_votes: int
    num_human_votes: int

class FeatureImportance(BaseModel):
    name: str
    importance: float
    family: str

class AnalysisResponse(BaseModel):
    file: FileInfo
    classical: ModelResult
    resnet18: ResNetResult
    ast: ModelResult
    consensus: ConsensusResult
    features: list  # [FeatureImportance]
    mel_spectrogram: Optional[list] = None  # 2-D list (rows × cols) for rendering
    inference_time_s: float


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _family(name: str) -> str:
    if "contrast" in name:   return "Spectral Contrast"
    if "centroid" in name:   return "Spectral Centroid"
    if "bandwidth" in name:  return "Spectral Bandwidth"
    if "rolloff" in name:    return "Spectral Rolloff"
    if "flatness" in name:   return "Spectral Flatness"
    if "mfcc" in name:       return "MFCC"
    if "chroma" in name:     return "Chroma"
    if "rms" in name:        return "Energy"
    if "zcr" in name:        return "Zero-Crossing Rate"
    if "tempo" in name:      return "Temporal"
    if "onset" in name:      return "Onset"
    if "harmonic" in name or "percussive" in name: return "Harmonic/Percussive"
    return "Other"


def _consensus(c_prob, r_prob, ast_prob,
               c_pred, r_pred, ast_pred) -> ConsensusResult:
    valid = [(p, pred) for p, pred in
             [(c_prob, c_pred), (r_prob, r_pred), (ast_prob, ast_pred)]
             if p is not None]
    if not valid:
        return ConsensusResult(
            status="ERROR",
            message="No method produced a valid result.",
            num_ai_votes=0, num_human_votes=0
        )

    probs = [p for p, _ in valid]
    preds = [pred for _, pred in valid]
    variance = float(np.var(probs))
    mean_prob = float(np.mean(probs))
    num_ai = preds.count("AI")
    num_human = preds.count("Human")

    if variance > 0.05 or (0.3 < mean_prob < 0.7):
        return ConsensusResult(
            status="INCONCLUSIVE",
            message=(
                "The detection methods disagree substantially on this audio. "
                "Treat the result cautiously rather than as a definitive classification."
            ),
            num_ai_votes=num_ai, num_human_votes=num_human
        )
    if num_ai == len(valid):
        return ConsensusResult(
            status="STRONG_CONSENSUS_AI",
            message=(
                f"All {len(valid)} methods indicate AI-generated patterns. "
                "Agreement between methods increases consistency of the observed evidence, "
                "but does not establish ground truth."
            ),
            num_ai_votes=num_ai, num_human_votes=num_human
        )
    if num_human == len(valid):
        return ConsensusResult(
            status="STRONG_CONSENSUS_HUMAN",
            message=(
                f"All {len(valid)} methods indicate human-composed patterns. "
                "Agreement between methods increases consistency of the observed evidence, "
                "but does not establish ground truth."
            ),
            num_ai_votes=num_ai, num_human_votes=num_human
        )
    majority = "AI" if num_ai > num_human else "Human"
    return ConsensusResult(
        status="MIXED",
        message=(
            f"Methods disagree. Majority ({max(num_ai,num_human)}/{len(valid)}) "
            f"suggests {majority}-generated. "
            "Note: Agreement does not guarantee correctness, especially for generators "
            "not represented during training."
        ),
        num_ai_votes=num_ai, num_human_votes=num_human
    )


def _run_classical(audio_path: str):
    try:
        preprocessor = _models["preprocessor"]
        extractor    = _models["classical_extractor"]
        rf           = _models["rf"]

        segments, meta = preprocessor.process(audio_path)
        if not segments:
            return None, None, None, "No audio segments"

        seg = segments[0]
        feats, feat_names = extractor.extract_features(seg, feature_set="B")
        X = feats.reshape(1, -1)
        prob = float(rf.predict_proba(X)[0][1])
        pred = "AI" if prob > 0.5 else "Human"
        return prob, pred, meta, None
    except Exception as e:
        logger.exception("Classical inference error")
        return None, None, None, str(e)


def _run_resnet(audio_path: str):
    try:
        preprocessor = _models["preprocessor"]
        extractor    = _models["resnet_extractor"]
        rmodels      = _models["resnet_models"]
        device       = _models["device"]

        segments, _ = preprocessor.process(audio_path)
        if not segments:
            return None, None, None, "No audio segments"

        seg = segments[0]
        mel = extractor.extract(seg).unsqueeze(0).to(device)

        results = []
        probs   = []
        for name, model in rmodels.items():
            with torch.no_grad():
                out  = model(mel)
                prob = float(torch.sigmoid(out).item())
                results.append({"name": name, "ai_probability": prob})
                probs.append(prob)

        avg_prob = float(np.mean(probs)) if probs else 0.5
        pred = "AI" if avg_prob > 0.5 else "Human"
        return avg_prob, pred, results, None
    except Exception as e:
        logger.exception("ResNet inference error")
        return None, None, [], str(e)


def _run_ast(audio_path: str):
    try:
        result   = _models["ast"].predict(audio_path)
        ai_prob  = float(result["ai_probability"])
        hum_prob = float(result["human_probability"])
        pred = "AI" if ai_prob > 0.5 else "Human"
        return ai_prob, hum_prob, pred, None
    except Exception as e:
        logger.exception("AST inference error")
        return None, None, None, str(e)


def _get_feature_importances(n: int = 10) -> list:
    importances = _models.get("rf_importances")
    if importances is None or not RF_FEATURE_NAMES or len(RF_FEATURE_NAMES) != len(importances):
        return []
    pairs = sorted(
        zip(RF_FEATURE_NAMES, importances),
        key=lambda x: x[1],
        reverse=True
    )[:n]
    return [
        {"name": nm, "importance": float(imp), "family": _family(nm)}
        for nm, imp in pairs
    ]


def _get_mel_image(audio_path: str) -> Optional[list]:
    """Returns mel-spectrogram as a 2-D list of floats (rows × cols) for the UI."""
    try:
        preprocessor = _models["preprocessor"]
        extractor    = _models["resnet_extractor"]
        segments, _  = preprocessor.process(audio_path)
        if not segments:
            return None
        mel = extractor.extract(segments[0])  # (1, 128, T)
        arr = mel.squeeze(0).cpu().numpy()    # (128, T)
        # downsample time axis to max 200 cols for JSON size
        if arr.shape[1] > 200:
            step = arr.shape[1] // 200
            arr  = arr[:, ::step]
        return arr.tolist()
    except Exception:
        logger.exception("Mel-spectrogram extraction error")
        return None


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(_models.get("device", "unknown")),
        "models_loaded": list(_models.keys()),
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    # Validate extension
    allowed = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(allowed))}"
        )

    # Save to temp file
    tmp_dir  = tempfile.mkdtemp(prefix="hs_")
    safe_name = f"upload_{uuid.uuid4().hex}{ext}"
    tmp_path = os.path.join(tmp_dir, safe_name)
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_size = os.path.getsize(tmp_path)
        t0 = time.time()

        # ── Run all three inference methods ──────────────────────────────────
        c_prob, c_pred, meta, c_err = _run_classical(tmp_path)
        r_prob, r_pred, r_results, r_err = _run_resnet(tmp_path)
        a_prob, a_hum, a_pred, a_err = _run_ast(tmp_path)

        inference_time = time.time() - t0

        # ── File info ─────────────────────────────────────────────────────────
        duration     = float(meta["duration"])     if meta else 0.0
        sample_rate  = int(meta["sample_rate"])    if meta else 16000
        num_segments = int(meta["num_segments"])   if meta else 0

        # ── Mel spectrogram ───────────────────────────────────────────────────
        mel_data = _get_mel_image(tmp_path)

        # ── Feature importances ───────────────────────────────────────────────
        top_features = _get_feature_importances(10)

        # ── Consensus ─────────────────────────────────────────────────────────
        consensus = _consensus(
            c_prob, r_prob, a_prob,
            c_pred if c_prob is not None else "Error",
            r_pred if r_prob is not None else "Error",
            a_pred if a_prob is not None else "Error",
        )

        return AnalysisResponse(
            file=FileInfo(
                name=file.filename or safe_name,
                size_bytes=file_size,
                duration=duration,
                sample_rate=sample_rate,
                num_segments=num_segments,
            ),
            classical=ModelResult(
                prediction=c_pred or "Error",
                ai_probability=c_prob or 0.0,
                human_probability=1.0 - (c_prob or 0.0),
                error=c_err,
            ),
            resnet18=ResNetResult(
                prediction=r_pred or "Error",
                ai_probability=r_prob or 0.0,
                human_probability=1.0 - (r_prob or 0.0),
                ensemble_size=len(r_results),
                checkpoint_results=r_results,
                error=r_err,
            ),
            ast=ModelResult(
                prediction=a_pred or "Error",
                ai_probability=a_prob or 0.0,
                human_probability=a_hum or 0.0,
                error=a_err,
            ),
            consensus=consensus,
            features=top_features,
            mel_spectrogram=mel_data,
            inference_time_s=round(inference_time, 2),
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
