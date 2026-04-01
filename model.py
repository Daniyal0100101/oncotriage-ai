"""
model.py — OncoTriage AI
NVIDIA NIM Evo 2 API client and RandomForest ranking head.
Research use only. Outputs are not clinically validated.
"""

import os
import io
import re
import base64
import time
import logging
import numpy as np
import requests
from typing import Optional, Dict, List, Tuple
import joblib

# ── API configuration ─────────────────────────────────────────────────────────

NIM_BASE_URL = "https://health.api.nvidia.com/v1"
EVO2_MODEL = "evo2-40b"
FORWARD_ENDPOINT = f"{NIM_BASE_URL}/biology/arc/{EVO2_MODEL}/forward"
GENERATE_ENDPOINT = f"{NIM_BASE_URL}/biology/arc/{EVO2_MODEL}/generate"

# Output layers for scoring (use final logits)
SCORING_LAYERS = ["output_layer"]

# Timeout in seconds
REQUEST_TIMEOUT = 120

# ── Feature flag ──────────────────────────────────────────────────────────────
# The hosted Evo 2 API (health.api.nvidia.com) does NOT expose the /forward
# logits endpoint — it raises "StripedHyena has no attribute output_layer".
# Keep False for the hosted API; set True only for private NIM deployments.
EVO2_USE_FORWARD_PASS = os.environ.get("EVO2_USE_FORWARD_PASS", "false").lower() == "true"

# Patterns that identify an unsupported / internal-error API response
_INTERNAL_ERROR_PATTERNS = [
    "StripedHyena has no attribute",
    "has no attribute",
    "Internal Server Error",
    "AttributeError",
]

_log = logging.getLogger(__name__)

_REDACTION_PATTERNS = [
    (r"(nvapi-[A-Za-z0-9_\-]+)", "[REDACTED_API_KEY]"),
    (r"(?i)\b(api[_-]?key|token|secret)\b\s*[:=]\s*['\"]?[^'\",\\s]+", r"\1=[REDACTED]"),
    (r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1[REDACTED]"),
    (r"(?i)([A-Z]:\\|/)[^\\/\r\n\t]+(?:[\\/][^\\/\r\n\t]+)+", "[REDACTED_PATH]"),
    (r"\b[ACGT]{20,}\b", "[REDACTED_SEQUENCE]"),
]


# ── Custom exception ──────────────────────────────────────────────────────────

class InferenceUnavailableError(RuntimeError):
    """Raised when the live Evo 2 API returns an internal / unsupported error.
    Callers should fall back to demo mode and surface a UI warning.
    """


# ── Evo 2 NIM Client ──────────────────────────────────────────────────────────

class Evo2Client:
    """Thin HTTP client for NVIDIA NIM Evo 2 endpoints."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _post(self, url: str, payload: dict) -> dict:
        """POST with debug logging and robust internal-error detection."""
        _log.debug("POST %s | payload keys: %s", url, list(payload.keys()))
        try:
            resp = self.session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            _log.debug(
                "Response status: %s | snippet: %s",
                resp.status_code,
                _sanitize_response_snippet(resp.text),
            )

            if not resp.ok:
                body_text = ""
                try:
                    body_text = resp.json().get("detail", resp.text[:400])
                except Exception:
                    body_text = resp.text[:400]

                _log.warning("Non-200 from %s — %s: %.200s", url, resp.status_code, body_text)

                if any(pat in body_text for pat in _INTERNAL_ERROR_PATTERNS):
                    raise InferenceUnavailableError(
                        f"Evo 2 API internal error (status {resp.status_code}): {body_text[:200]}"
                    )
                if resp.status_code == 401:
                    raise RuntimeError("❌ Invalid API key. Please check your NVIDIA NIM key (nvapi-...).")
                elif resp.status_code == 422:
                    raise RuntimeError(f"❌ Invalid request payload: {body_text}")
                elif resp.status_code == 429:
                    raise RuntimeError("⚠️ Rate limit exceeded. Free tier allows limited API calls. Wait and retry.")
                else:
                    raise RuntimeError(f"API error {resp.status_code}: {body_text}")

            try:
                data = resp.json()
            except Exception:
                raise RuntimeError("Could not decode API response as JSON.")

            # Detect internal errors embedded inside a 200 response body
            error_field = str(data.get("error") or data.get("message") or "")
            if error_field and any(pat in error_field for pat in _INTERNAL_ERROR_PATTERNS):
                _log.warning("Internal error in 200 body: %.200s", error_field)
                raise InferenceUnavailableError(
                    f"Evo 2 API internal error in response body: {error_field[:200]}"
                )

            return data

        except requests.exceptions.Timeout:
            raise RuntimeError(
                "⏱ Request timed out (>120s). The NIM API may be cold-starting or overloaded. "
                "Try reducing sequence length or retry in a moment."
            )
        except (InferenceUnavailableError, RuntimeError):
            raise
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error contacting NVIDIA API: {e}")

    def forward(self, sequence: str,
                output_layers: List[str] = None) -> Tuple[Optional[np.ndarray], float]:
        """
        Run model forward pass.
        Returns (logits_array, elapsed_ms).
        logits_array shape: [seq_len, 512] (output_layer).
        """
        if output_layers is None:
            output_layers = SCORING_LAYERS

        payload = {
            "sequence": sequence.upper(),
            "output_layers": output_layers,
        }
        result = self._post(FORWARD_ENDPOINT, payload)

        data_b64 = result.get("data")
        elapsed_ms = result.get("elapsed_ms", 0)

        if data_b64:
            logits = _decode_npz(data_b64)
        else:
            logits = None

        return logits, elapsed_ms

    def generate(self, sequence: str, num_tokens: int = 100,
                 temperature: float = 0.7, top_k: int = 4,
                 enable_logits: bool = False) -> dict:
        """
        Generate DNA sequence continuation from a prompt.
        Returns full API response dict.
        """
        if not self.api_key:
            # DEMO MODE
            import random
            time.sleep(1.0)
            mock_seq = "".join(random.choices("ACGT", k=num_tokens))
            return {"sequence": mock_seq, "elapsed_ms": 450}
            
        payload = {
            "sequence": sequence.upper(),
            "num_tokens": num_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "enable_logits": enable_logits,
            "enable_sampled_probs": True,
        }
        return self._post(GENERATE_ENDPOINT, payload)

    def score_variant(self, ref_seq: str, alt_seq: str) -> Dict:
        """
        Compute Δlog-likelihood between ref and alt sequences.
        Higher negative delta → more damaging.

        Routing:
          - No API key          → demo mode (clearly labeled).
          - EVO2_USE_FORWARD_PASS=False (default) → /generate endpoint.
          - EVO2_USE_FORWARD_PASS=True  → /forward endpoint (private NIM only).

        Returns dict with delta_ll, ll_ref, ll_alt, elapsed_ms_total, demo_mode.
        Raises InferenceUnavailableError on internal API failures so the caller
        can fall back gracefully.
        """
        if not self.api_key:
            return _demo_score_result()

        if not EVO2_USE_FORWARD_PASS:
            return self._score_via_generate(ref_seq, alt_seq)

        # ── Forward-pass path (private NIM deployments only) ──────────────────
        t0 = time.time()
        logits_ref, ms_ref = self.forward(ref_seq)
        logits_alt, ms_alt = self.forward(alt_seq)
        elapsed_total = int((time.time() - t0) * 1000)

        if logits_ref is None or logits_alt is None:
            raise RuntimeError(
                "Could not decode model logits from API response. "
                "Check API key and sequence validity."
            )

        ll_ref = _compute_ll(logits_ref, ref_seq)
        ll_alt = _compute_ll(logits_alt, alt_seq)
        delta_ll = ll_alt - ll_ref

        return {
            "ll_ref": ll_ref, "ll_alt": ll_alt, "delta_ll": delta_ll,
            "elapsed_ms_ref": ms_ref, "elapsed_ms_alt": ms_alt,
            "elapsed_ms_total": elapsed_total, "demo_mode": False,
        }

    def _score_via_generate(self, ref_seq: str, alt_seq: str) -> Dict:
        """
        Derive a proxy Δlog-likelihood using the /generate endpoint.
        Requests 1 token with enable_logits + enable_sampled_probs.
        Raises InferenceUnavailableError if the endpoint is unsupported.
        """
        t0 = time.time()

        def _make_payload(seq: str) -> dict:
            return {
                "sequence": seq.upper(),
                "num_tokens": 1,
                "top_k": 1,
                "enable_logits": True,
                "enable_sampled_probs": True,
            }

        ref_result = self._post(GENERATE_ENDPOINT, _make_payload(ref_seq))
        alt_result = self._post(GENERATE_ENDPOINT, _make_payload(alt_seq))
        elapsed_total = int((time.time() - t0) * 1000)

        ll_ref = _extract_ll_from_generate(ref_result)
        ll_alt = _extract_ll_from_generate(alt_result)
        delta_ll = ll_alt - ll_ref

        return {
            "ll_ref": ll_ref, "ll_alt": ll_alt, "delta_ll": delta_ll,
            "elapsed_ms_ref": 0, "elapsed_ms_alt": 0,
            "elapsed_ms_total": elapsed_total, "demo_mode": False,
        }


# ── Demo fallback + generate LL extraction ───────────────────────────────────

def _demo_score_result() -> Dict:
    """Return a clearly-labeled demo-mode score dict. No live API call made."""
    import random
    time.sleep(0.5)
    ll_ref = -2.0 - random.random()
    delta_ll = -0.5 - random.random() * 2.5
    ll_alt = ll_ref + delta_ll
    return {
        "ll_ref": ll_ref, "ll_alt": ll_alt, "delta_ll": delta_ll,
        "elapsed_ms_ref": 120, "elapsed_ms_alt": 115, "elapsed_ms_total": 235,
        "demo_mode": True,
    }


def _extract_ll_from_generate(gen_result: dict) -> float:
    """
    Extract a log-likelihood proxy from a /generate response dict.
    Prefers sampled_probs; falls back to logits blob; else returns neutral -2.0.
    """
    import math
    sampled = gen_result.get("sampled_probs")
    if sampled and isinstance(sampled, list):
        log_probs = [math.log(max(float(p), 1e-12)) for p in sampled if p is not None]
        if log_probs:
            return float(sum(log_probs) / len(log_probs))

    logits_b64 = gen_result.get("logits")
    if logits_b64:
        arr = _decode_npz(logits_b64)
        if arr is not None:
            seq = gen_result.get("sequence")
            if not isinstance(seq, str) or not seq:
                return -2.0
            expected_tokens = int(arr.shape[0]) if getattr(arr, "ndim", 0) >= 1 else 0
            if expected_tokens <= 0 or len(seq) != expected_tokens:
                return -2.0
            return _compute_ll(arr, seq)

    return -2.0


def _sanitize_response_snippet(text: Optional[str], limit: int = 100) -> str:
    """Redact sensitive fields from API responses before debug logging."""
    if not text:
        return ""

    sanitized = str(text).replace("\r", " ").replace("\n", " ")
    for pattern, replacement in _REDACTION_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized)
    sanitized = " ".join(sanitized.split())
    return sanitized[:limit]


# ── Logit decoding ────────────────────────────────────────────────────────────

def _decode_npz(b64_data: str) -> Optional[np.ndarray]:
    """Decode Base64-encoded NPZ blob → numpy array."""
    try:
        raw = base64.b64decode(b64_data)
        buf = io.BytesIO(raw)
        npz = np.load(buf, allow_pickle=False)
        # output_layer: [seq_len, batch_size, 512] or [seq_len, 512]
        key = "output_layer" if "output_layer" in npz.files else npz.files[0]
        arr = npz[key]
        if arr.ndim == 3:
            arr = arr[:, 0, :]  # take batch 0
        return arr
    except Exception:
        return None


def _compute_ll(logits: np.ndarray, sequence: str) -> float:
    """Per-token mean log-likelihood for a DNA sequence given logits."""
    seq = sequence.upper()
    # ASCII indices for ACGT in the Evo 2 vocab-512 mapping
    base_ascii = {ord("A"), ord("C"), ord("T"), ord("G")}   # 65,67,84,71
    # Softmax
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(shifted.astype(np.float64))
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

    log_probs = []
    for i, base in enumerate(seq):
        if i >= logits.shape[0]:
            break
        idx = ord(base)
        if idx in base_ascii and idx < probs.shape[-1]:
            log_probs.append(np.log(probs[i, idx] + 1e-12))

    return float(np.mean(log_probs)) if log_probs else 0.0


# ── ML Scoring Head ───────────────────────────────────────────────────────────

ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), "brca_classifier.pkl")


class BRCAScorer:
    """
    Lightweight RandomForest head trained on ClinVar BRCA1/BRCA2 features.
    Falls back to a heuristic score from Δlog-likelihood alone if no model file.
    """

    def __init__(self):
        self.model: Optional[object] = None
        self._try_load()

    def _try_load(self):
        if os.path.exists(ML_MODEL_PATH):
            try:
                self.model = joblib.load(ML_MODEL_PATH)
            except Exception:
                self.model = None

    def get_features(self, delta_ll: float, ll_ref: float, ll_alt: float,
                     ref_seq: str, alt_seq: str, annotation: str = "") -> np.ndarray:
        """Compose feature vector for the ML classifier."""
        seq_len = len(ref_seq)
        gc_ref = (ref_seq.count("G") + ref_seq.count("C")) / max(seq_len, 1)
        gc_alt = (alt_seq.count("G") + alt_seq.count("C")) / max(len(alt_seq), 1)
        at_skew = (ref_seq.count("A") - ref_seq.count("T")) / max(seq_len, 1)
        has_splice = 1 if "splice" in annotation.lower() else 0
        has_stop = 1 if "stop" in annotation.lower() else 0

        features = np.array([
            delta_ll,
            ll_ref,
            ll_alt,
            abs(delta_ll),
            gc_ref,
            gc_alt,
            gc_alt - gc_ref,
            at_skew,
            seq_len,
            has_splice,
            has_stop,
            float(delta_ll < -0.5),
            float(delta_ll < -1.0),
        ]).reshape(1, -1)
        return features

    def predict_risk(self, delta_ll: float, ll_ref: float, ll_alt: float,
                     ref_seq: str = "", alt_seq: str = "",
                     annotation: str = "") -> float:
        """
        Compute a variant review priority score in [0, 1].
        Higher score = higher review priority signal (not a clinical diagnosis).
        Uses trained RF model if available, otherwise a calibrated heuristic.
        """
        if self.model is not None:
            feats = self.get_features(delta_ll, ll_ref, ll_alt, ref_seq, alt_seq, annotation)
            try:
                prob = self.model.predict_proba(feats)[0, 1]
                return float(prob)
            except Exception:
                pass

        # ── Heuristic fallback ───────────────────────────────────────────────
        # Δ LL is negative and large → more damaging.
        # Empirically, most pathogenic BRCA variants have Δ < -0.5.
        return _heuristic_risk(delta_ll, annotation)

    def train_on_clinvar(self, training_data: List[Dict]) -> Dict:
        """
        Train the RandomForest on ClinVar features.
        training_data: list of {"delta_ll", "ll_ref", "ll_alt",
                                "ref_seq", "alt_seq", "label" (1=pathogenic, 0=benign)}
        Returns training metrics.
        """
        if not training_data:
            return {"error": "No training data provided."}

        X_rows, y = [], []
        for item in training_data:
            feats = self.get_features(
                item["delta_ll"], item.get("ll_ref", 0.0), item.get("ll_alt", 0.0),
                item.get("ref_seq", ""), item.get("alt_seq", ""),
                item.get("annotation", ""),
            )
            X_rows.append(feats[0])
            y.append(item["label"])

        X = np.array(X_rows)
        y = np.array(y)

        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
        pipeline.fit(X, y)

        joblib.dump(pipeline, ML_MODEL_PATH)
        self.model = pipeline

        return {
            "auc_cv_mean": float(cv_scores.mean()),
            "auc_cv_std": float(cv_scores.std()),
            "n_samples": len(y),
            "n_pathogenic": int(y.sum()),
            "n_benign": int((1 - y).sum()),
        }


def _heuristic_risk(delta_ll: float, annotation: str = "") -> float:
    """
    Map Δlog-likelihood to a review priority score via a sigmoid-like calibration.
    Empirical observation: higher-impact variants tend to cluster at Δ < -1.0.
    This is a heuristic approximation, not a validated clinical classifier.
    """
    # Splice/stop override
    if "splice" in annotation.lower():
        base = 0.85
    elif "stop" in annotation.lower():
        base = 0.80
    else:
        # sigmoid centred at Δ = -0.5
        import math
        x = -(delta_ll + 0.5) * 2.5
        base = 1.0 / (1.0 + math.exp(-x))

    return float(max(0.0, min(1.0, base)))


# ── Convenience: build synthetic training data ────────────────────────────────

def build_synthetic_training_data(n_pathogenic: int = 200,
                                   n_benign: int = 200) -> List[Dict]:
    """
    Build a SYNTHETIC training set for offline / demo use only.

    WARNING: This function generates simulated data from statistical distributions.
    It does NOT use real ClinVar records or real Evo 2 scores.
    The RF ranking head trained on this data is NOT scientifically validated.
    Replace with real ClinVar-derived Evo 2 scores before any research claims.
    Real data can be fetched via clinvar_data.py.
    """
    rng = np.random.default_rng(42)
    data = []

    for _ in range(n_pathogenic):
        delta = rng.normal(-1.5, 0.6)
        data.append({
            "delta_ll": delta, "ll_ref": rng.normal(-2, 0.3),
            "ll_alt": rng.normal(-2, 0.3) + delta,
            "ref_seq": "", "alt_seq": "", "annotation": "",
            "label": 1,
        })

    for _ in range(n_benign):
        delta = rng.normal(0.1, 0.4)
        data.append({
            "delta_ll": delta, "ll_ref": rng.normal(-1.8, 0.3),
            "ll_alt": rng.normal(-1.8, 0.3) + delta,
            "ref_seq": "", "alt_seq": "", "annotation": "",
            "label": 0,
        })

    rng.shuffle(data)
    return data
