"""
test_app.py — OncoTriage AI

Unit tests for variant parsing, scoring helpers, and data utilities.
Run: python -m pytest test_app.py -v
"""

import sys
import os
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    parse_variant,
    validate_dna_sequence,
    apply_variant_to_sequence,
    annotate_variant,
    compute_sequence_log_likelihood,
    compute_likelihood_delta,
    _risk_color,
    risk_label,
    results_to_csv,
)
from model import (
    _decode_npz,
    _compute_ll,
    _heuristic_risk,
    BRCAScorer,
    build_synthetic_training_data,
)
from clinvar_data import (
    get_demo_sequence,
)


# ── Variant parsing ────────────────────────────────────────────────────────────

class TestParseVariant:
    def test_cdna_hgvs(self):
        v = parse_variant("c.5123C>A")
        assert v is not None
        assert v["type"] == "cdna"
        assert v["ref"] == "C"
        assert v["alt"] == "A"

    def test_cdna_hgvs_lowercase(self):
        v = parse_variant("c.185g>a")
        assert v is not None
        assert v["ref"] == "G"
        assert v["alt"] == "A"

    def test_genomic_hgvs(self):
        v = parse_variant("g.41246481C>T")
        assert v is not None
        assert v["type"] == "genomic"
        assert v["position"] == 41246481

    def test_positional_space(self):
        v = parse_variant("185 A>G")
        assert v is not None
        assert v["position"] == 185
        assert v["ref"] == "A"
        assert v["alt"] == "G"

    def test_invalid_variant(self):
        assert parse_variant("not_a_variant") is None
        assert parse_variant("") is None
        assert parse_variant("c.ABCX>Y") is None

    def test_intronic_position(self):
        v = parse_variant("c.5946+1G>T")
        assert v is not None  # should match the base pattern

    def test_synonymous_detected(self):
        v = parse_variant("c.100A>A")
        # ref == alt is valid to parse but app flags separately
        assert v is not None
        assert v["ref"] == v["alt"]


# ── Sequence validation ────────────────────────────────────────────────────────

class TestValidateDNASequence:
    def test_valid(self):
        ok, msg = validate_dna_sequence("ATCGATCGATCG")
        assert ok, msg

    def test_too_short(self):
        ok, msg = validate_dna_sequence("ATCG")
        assert not ok
        assert "short" in msg.lower()

    def test_invalid_chars(self):
        ok, msg = validate_dna_sequence("ATCGNNNNXXXX")
        assert not ok
        assert "invalid" in msg.lower()

    def test_lowercase_normalised(self):
        ok, msg = validate_dna_sequence("atcgatcgatcg")
        assert ok

    def test_empty(self):
        ok, msg = validate_dna_sequence("")
        assert not ok


# ── Sequence mutation ──────────────────────────────────────────────────────────

class TestApplyVariant:
    SEQ = "ATCGATCGATCGATCGATCGATCG"

    def test_simple_snv(self):
        parsed = {"type": "positional", "position": 5, "ref": "A", "alt": "T"}
        ref_ctx, alt_ctx = apply_variant_to_sequence(self.SEQ, parsed, 20)
        assert len(ref_ctx) <= 20
        assert len(alt_ctx) <= 20
        assert ref_ctx != alt_ctx or parsed["ref"] == parsed["alt"]

    def test_context_clipped(self):
        parsed = {"type": "positional", "position": 1, "ref": "A", "alt": "G"}
        ref_ctx, alt_ctx = apply_variant_to_sequence(self.SEQ, parsed, 512)
        # Cannot exceed original sequence length
        assert len(ref_ctx) <= len(self.SEQ)

    def test_cdna_variant(self):
        parsed = parse_variant("c.5C>T")
        ref_ctx, alt_ctx = apply_variant_to_sequence(self.SEQ, parsed, 20)
        assert isinstance(ref_ctx, str)
        assert isinstance(alt_ctx, str)


# ── Annotation ─────────────────────────────────────────────────────────────────

class TestAnnotateVariant:
    def test_transition(self):
        parsed = {"ref": "A", "alt": "G", "position": "100"}
        ann = annotate_variant(parsed)
        assert "Transition" in ann

    def test_transversion(self):
        parsed = {"ref": "A", "alt": "C", "position": "100"}
        ann = annotate_variant(parsed)
        assert "Transversion" in ann

    def test_splice_site(self):
        parsed = {"ref": "G", "alt": "A", "position": "c.1234+1"}
        ann = annotate_variant(parsed)
        assert "splice" in ann.lower()

    def test_intronic_near(self):
        parsed = {"ref": "T", "alt": "A", "position": "c.500+5"}
        ann = annotate_variant(parsed)
        assert "intronic" in ann.lower() or "Intronic" in ann or "Transition" in ann


# ── Log-likelihood computation ─────────────────────────────────────────────────

class TestLogLikelihood:
    def _make_logits(self, seq_len: int, favor_base: str = "A") -> np.ndarray:
        """Make dummy logits that strongly favor one base."""
        logits = np.zeros((seq_len, 512), dtype=np.float32)
        base_idx = ord(favor_base)
        logits[:, base_idx] = 10.0  # strong signal
        return logits

    def test_high_ll_for_correct_base(self):
        logits = self._make_logits(5, "A")
        seq = "AAAAA"
        ll = compute_sequence_log_likelihood(logits, seq)
        assert ll > -0.05, f"Expected near-0 LL for confident prediction, got {ll}"

    def test_low_ll_for_wrong_base(self):
        logits = self._make_logits(5, "A")
        seq = "CCCCC"
        ll = compute_sequence_log_likelihood(logits, seq)
        assert ll < -3.0, f"Expected low LL for mis-predicted sequence, got {ll}"

    def test_delta_negative_for_damaging(self):
        logits_ref = self._make_logits(5, "A")
        logits_alt = self._make_logits(5, "C")  # alt doesn't match ref
        delta = compute_likelihood_delta(logits_ref, "AAAAA", logits_alt, "CAAAA")
        assert delta < 0, f"Damaging variant should have negative delta, got {delta}"

    def test_compute_ll_model_module(self):
        logits = self._make_logits(4, "G")
        ll = _compute_ll(logits, "GGGG")
        assert ll > -0.05

    def test_empty_sequence(self):
        logits = self._make_logits(5, "A")
        ll = compute_sequence_log_likelihood(logits, "")
        assert ll == 0.0


# ── Heuristic risk scoring ────────────────────────────────────────────────────

class TestHeuristicRisk:
    def test_strongly_negative_delta_high_risk(self):
        risk = _heuristic_risk(-3.0)
        assert risk > 0.85

    def test_positive_delta_low_risk(self):
        risk = _heuristic_risk(1.0)
        assert risk < 0.2

    def test_splice_site_override(self):
        risk = _heuristic_risk(0.5, "Located at canonical splice site")
        assert risk >= 0.85

    def test_risk_in_range(self):
        for delta in np.linspace(-5, 5, 20):
            r = _heuristic_risk(float(delta))
            assert 0.0 <= r <= 1.0


# ── Risk labels and colors ────────────────────────────────────────────────────

class TestRiskHelpers:
    def test_high_risk_label(self):
        assert "HIGH" in risk_label(0.90)

    def test_benign_label(self):
        assert "BENIGN" in risk_label(0.05).upper()

    def test_uncertain_label(self):
        assert "UNCERTAIN" in risk_label(0.50).upper() or "VUS" in risk_label(0.50)

    def test_colors(self):
        assert _risk_color(0.90) == "#e74c3c"
        assert _risk_color(0.50) == "#f39c12"
        assert _risk_color(0.10) == "#2ecc71"


# ── ML Scorer ─────────────────────────────────────────────────────────────────

class TestBRCAScorer:
    def test_heuristic_fallback(self):
        scorer = BRCAScorer()
        # Without a trained model, uses heuristic
        risk = scorer.predict_risk(-2.0, -2.0, -4.0)
        assert 0.0 <= risk <= 1.0

    def test_train_synthetic(self):
        scorer = BRCAScorer()
        data = build_synthetic_training_data(50, 50)
        metrics = scorer.train_on_clinvar(data)
        assert "auc_cv_mean" in metrics
        assert metrics["auc_cv_mean"] > 0.5, "AUC should be above random chance"

    def test_trained_model_predicts(self):
        scorer = BRCAScorer()
        data = build_synthetic_training_data(100, 100)
        scorer.train_on_clinvar(data)
        # High-delta variant should score higher risk
        risk_high = scorer.predict_risk(-3.0, -2.0, -5.0)
        risk_low = scorer.predict_risk(1.0, -1.5, -0.5)
        assert risk_high > risk_low, (
            f"High damage (Δ=-3.0) should score higher than low damage (Δ=+1.0). "
            f"Got {risk_high:.3f} vs {risk_low:.3f}"
        )

    def test_feature_vector_shape(self):
        scorer = BRCAScorer()
        feats = scorer.get_features(-1.0, -2.0, -3.0, "ATCG", "TTCG")
        assert feats.shape == (1, 13)


# ── Synthetic training data ───────────────────────────────────────────────────

class TestSyntheticData:
    def test_build_data(self):
        data = build_synthetic_training_data(30, 30)
        assert len(data) == 60
        labels = [d["label"] for d in data]
        assert 1 in labels and 0 in labels

    def test_pathogenic_delta_distribution(self):
        data = build_synthetic_training_data(200, 200)
        path_deltas = [d["delta_ll"] for d in data if d["label"] == 1]
        benign_deltas = [d["delta_ll"] for d in data if d["label"] == 0]
        assert np.mean(path_deltas) < np.mean(benign_deltas), (
            "Pathogenic variants should have lower mean Δ LL than benign"
        )


# ── Demo sequences ────────────────────────────────────────────────────────────

class TestDemoSequences:
    def test_brca1_demo(self):
        seq = get_demo_sequence("BRCA1")
        assert len(seq) >= 50
        assert all(c in "ACGT" for c in seq.upper())

    def test_brca2_demo(self):
        seq = get_demo_sequence("BRCA2")
        assert len(seq) >= 50

    def test_unknown_gene_fallback(self):
        seq = get_demo_sequence("UNKNOWNGENE")
        assert len(seq) > 0


# ── CSV export ────────────────────────────────────────────────────────────────

class TestCSVExport:
    def test_export_not_empty(self):
        records = [
            {"variant": "c.185G>A", "risk_score": 0.82, "classification": "High Risk"},
            {"variant": "c.100A>G", "risk_score": 0.12, "classification": "Benign"},
        ]
        csv_bytes = results_to_csv(records)
        assert len(csv_bytes) > 0
        text = csv_bytes.decode("utf-8")
        assert "variant" in text
        assert "c.185G>A" in text


# ── NPZ decode (no real API needed) ──────────────────────────────────────────

class TestNPZDecode:
    def test_invalid_b64_returns_none(self):
        result = _decode_npz("not_valid_base64!!!")
        assert result is None

    def test_empty_b64_returns_none(self):
        result = _decode_npz("")
        assert result is None

    def test_valid_npz(self):
        import io, base64
        arr = np.zeros((10, 512), dtype=np.float32)
        buf = io.BytesIO()
        np.savez(buf, output_layer=arr)
        b64 = base64.b64encode(buf.getvalue()).decode()
        decoded = _decode_npz(b64)
        assert decoded is not None
        assert decoded.shape == (10, 512)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(__file__) or ".",
    )
    sys.exit(result.returncode)
