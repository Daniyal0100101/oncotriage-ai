"""
utils.py — OncoTriage AI

Variant parsing, sequence mutation, and visualization helpers.
"""

import re
import io
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Tuple, Dict

# ── Genomics constants ────────────────────────────────────────────────────────

COMPLEMENT = str.maketrans("ACTGactg", "TGACTGac")
BASES = set("ACGT")

# Known functional annotations for contextual explanation
SPLICE_DONOR_REGEX = re.compile(r"[Gg][Tt]")
SPLICE_ACCEPTOR_REGEX = re.compile(r"[Aa][Gg]$")

CLINVAR_SIGNIFICANCE_COLORS = {
    "Pathogenic": "#e74c3c",
    "Likely pathogenic": "#e67e22",
    "Uncertain significance": "#f1c40f",
    "Likely benign": "#27ae60",
    "Benign": "#2ecc71",
}

# ── HGVS variant parsing ──────────────────────────────────────────────────────

HGVS_CDNA_PATTERN = re.compile(
    r"c\.(-?\d+(?:[+-]\d+)?)([ACGT])>([ACGT])", re.IGNORECASE
)
HGVS_GENOMIC_PATTERN = re.compile(
    r"g\.(\d+)([ACGT])>([ACGT])", re.IGNORECASE
)
POSITIONAL_PATTERN = re.compile(
    r"^(\d+)\s+([ACGT])\s*[->/]\s*([ACGT])$", re.IGNORECASE
)


def parse_variant(variant_str: str) -> Optional[Dict]:
    """
    Parse a variant string into a structured dict.
    Supports: c.5123C>A, g.41246481C>T, '5123 C>A', '5123 C A'
    Returns None if unparseable.
    """
    v = variant_str.strip()

    # cDNA HGVS
    m = HGVS_CDNA_PATTERN.search(v)
    if m:
        return {
            "type": "cdna",
            "position": m.group(1),
            "ref": m.group(2).upper(),
            "alt": m.group(3).upper(),
            "notation": v,
        }

    # Genomic HGVS
    m = HGVS_GENOMIC_PATTERN.search(v)
    if m:
        return {
            "type": "genomic",
            "position": int(m.group(1)),
            "ref": m.group(2).upper(),
            "alt": m.group(3).upper(),
            "notation": v,
        }

    # Simple positional
    m = POSITIONAL_PATTERN.match(v)
    if m:
        return {
            "type": "positional",
            "position": int(m.group(1)),
            "ref": m.group(2).upper(),
            "alt": m.group(3).upper(),
            "notation": v,
        }

    return None


def validate_dna_sequence(seq: str) -> Tuple[bool, str]:
    """Return (is_valid, error_message)."""
    seq = seq.strip().upper()
    if not seq:
        return False, "Sequence cannot be empty."
    if len(seq) < 10:
        return False, "Sequence too short (minimum 10 bp)."
    if len(seq) > 100_000:
        return False, "Sequence too long (max 100,000 bp for NIM API)."
    invalid = set(seq) - BASES
    if invalid:
        return False, f"Invalid characters in sequence: {', '.join(sorted(invalid))}"
    return True, ""


def apply_variant_to_sequence(sequence: str, variant: Dict, context_window: int = 2048) -> Tuple[str, str]:
    """
    Given a long sequence and a parsed variant, return (ref_context, alt_context)
    centred on the variant position, clipped to context_window.
    For cDNA variants the position is 1-based from the start of the provided sequence.
    """
    seq = sequence.strip().upper()
    pos_raw = variant.get("position", "1")

    # Convert position to 0-based int
    try:
        pos = int(str(pos_raw).lstrip("c.").lstrip("g.").split("+")[0].split("-")[0]) - 1
    except (ValueError, AttributeError):
        pos = 0

    pos = max(0, min(pos, len(seq) - 1))

    ref = variant.get("ref", "")
    alt = variant.get("alt", "")

    # Build alt sequence
    alt_seq = seq[:pos] + alt + seq[pos + len(ref):]

    # Extract context window
    half = context_window // 2
    start = max(0, pos - half)
    end = min(len(seq), pos + half)

    ref_ctx = seq[start:end]
    alt_ctx = alt_seq[start:end]

    return ref_ctx, alt_ctx


def reverse_complement(seq: str) -> str:
    return seq.upper().translate(COMPLEMENT)[::-1]


# ── Logit → per-base probability ─────────────────────────────────────────────

BASE_INDICES = {65: "A", 67: "C", 84: "T", 71: "G"}  # ASCII → base


def decode_npz_logits(b64_npz: str) -> Optional[np.ndarray]:
    """Decode a Base64-encoded NPZ blob and return the output_layer array."""
    try:
        raw = base64.b64decode(b64_npz)
        buf = io.BytesIO(raw)
        npz = np.load(buf, allow_pickle=False)
        # output_layer shape: [seq_len, batch, 512]
        arr = npz.get("output_layer", None)
        if arr is None:
            # Try first available key
            key = list(npz.files)[0]
            arr = npz[key]
        return arr
    except Exception:
        return None


def compute_sequence_log_likelihood(logits: np.ndarray, sequence: str) -> float:
    """
    Compute mean per-token log-likelihood for the given sequence.
    logits: [seq_len, batch, vocab_size] or [seq_len, vocab_size]
    sequence: DNA bases at each position
    """
    if logits.ndim == 3:
        logits = logits[:, 0, :]  # take batch 0
    # Softmax
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(shifted)
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

    seq = sequence.upper()
    log_probs = []
    ascii_to_idx = {65: 0, 67: 1, 84: 2, 71: 3}  # A,C,T,G vocab positions

    for i, base in enumerate(seq):
        if i >= logits.shape[0]:
            break
        b_ascii = ord(base)
        if b_ascii in BASE_INDICES:
            vocab_idx = b_ascii  # direct ASCII index into vocab-512
            if vocab_idx < probs.shape[-1]:
                p = probs[i, vocab_idx]
                log_probs.append(np.log(p + 1e-12))

    return float(np.mean(log_probs)) if log_probs else 0.0


def compute_likelihood_delta(ref_logits: np.ndarray, ref_seq: str,
                             alt_logits: np.ndarray, alt_seq: str) -> float:
    """
    Δlog-likelihood = LL(alt) − LL(ref).
    Negative Δ → sequence is less likely under the model → more damaging.
    """
    ll_ref = compute_sequence_log_likelihood(ref_logits, ref_seq)
    ll_alt = compute_sequence_log_likelihood(alt_logits, alt_seq)
    return ll_alt - ll_ref


# ── Functional annotation ─────────────────────────────────────────────────────

def annotate_variant(variant: Dict, sequence_context: str = "") -> str:
    """
    Generate a plain-language annotation for the variant.
    """
    ref = variant.get("ref", "")
    alt = variant.get("alt", "")
    pos = str(variant.get("position", ""))

    explanations = []

    # Transition vs transversion
    transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    if (ref, alt) in transitions:
        explanations.append("Transition substitution (common type)")
    else:
        explanations.append("Transversion substitution (less common, often more impactful)")

    # Stop-codon check (naive — real check needs reading frame)
    stop_codons = {"TAA", "TAG", "TGA"}
    if sequence_context:
        ctx_upper = sequence_context.upper()
        if ctx_upper[len(ctx_upper)//2 - 1: len(ctx_upper)//2 + 2] in stop_codons:
            explanations.append("⚠ Possible premature stop codon in surrounding context")

    # Splice site proximity (positions ending in +1, +2, -1, -2 are canonical splice sites)
    pos_str = str(pos)
    if any(pos_str.endswith(s) for s in ["+1", "+2", "-1", "-2"]):
        explanations.append("🔴 Located at canonical splice site — likely disrupts splicing")
    elif any(s in pos_str for s in ["+", "-"]):
        explanations.append("Located in intronic region near exon boundary")

    return " | ".join(explanations) if explanations else "Standard coding substitution"


# ── Visualisations ────────────────────────────────────────────────────────────

def make_impact_bar_chart(delta_ll: float, risk_score: float) -> go.Figure:
    """
    Horizontal bar chart showing Δlog-likelihood and risk score.
    """
    categories = ["Likelihood Δ (normalised)", "Risk Score"]
    # Normalise delta_ll to [0, 1] roughly (clamp between -5 and 5)
    delta_norm = min(1.0, max(0.0, 1 - (delta_ll + 5) / 10))
    values = [delta_norm, risk_score]
    colors = [
        _risk_color(delta_norm),
        _risk_color(risk_score),
    ]
    labels = [f"{delta_ll:.4f}", f"{risk_score:.3f}"]

    fig = go.Figure()
    for i, (cat, val, col, lab) in enumerate(zip(categories, values, colors, labels)):
        fig.add_trace(go.Bar(
            x=[val], y=[cat], orientation="h",
            marker_color=col,
            text=lab, textposition="outside",
            name=cat, showlegend=False,
        ))

    fig.update_layout(
        xaxis=dict(range=[0, 1.15], title="Score (0 = lower priority, 1 = higher priority)"),
        yaxis_title="",
        height=200,
        margin=dict(l=20, r=40, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.6)",
        font=dict(color="#e0e0e0"),
        title=dict(text="Variant Impact Scores", font=dict(size=14)),
    )
    return fig


def make_genome_browser_snippet(sequence: str, position: int, ref: str, alt: str,
                                 window: int = 40) -> go.Figure:
    """
    A simple genome-browser-style sequence viewer around the variant site.
    """
    seq = sequence.upper()
    center = min(max(position - 1, 0), len(seq) - 1)
    start = max(0, center - window // 2)
    end = min(len(seq), center + window // 2)
    snippet = seq[start:end]

    colors = []
    for i, base in enumerate(snippet):
        global_pos = start + i
        if global_pos == center:
            colors.append("#e74c3c")  # highlight variant
        elif base == "A":
            colors.append("#3498db")
        elif base == "C":
            colors.append("#2ecc71")
        elif base == "G":
            colors.append("#f39c12")
        elif base == "T":
            colors.append("#9b59b6")
        else:
            colors.append("#95a5a6")

    x_pos = list(range(start, start + len(snippet)))
    hover_text = [
        f"Pos: {x+1}<br>Ref: {b}<br>{'← VARIANT SITE' if x == center else ''}"
        for x, b in zip(x_pos, snippet)
    ]

    fig = go.Figure(go.Bar(
        x=x_pos,
        y=[1] * len(snippet),
        text=list(snippet),
        textposition="inside",
        marker_color=colors,
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False,
    ))

    # Mark variant site
    fig.add_vline(x=center, line_dash="dash", line_color="#e74c3c",
                  annotation_text=f"{ref}→{alt}", annotation_position="top")

    fig.update_layout(
        title=f"Sequence Context (pos {start+1}–{end})",
        xaxis_title="Genomic Position",
        yaxis_visible=False,
        height=180,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.6)",
        font=dict(color="#e0e0e0"),
    )
    return fig


def make_risk_gauge(risk_score: float) -> go.Figure:
    """Gauge chart showing final risk probability."""
    color = _risk_color(risk_score)
    label = risk_label(risk_score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(risk_score * 100, 1),
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": f"Review Priority Score<br><b>{label}</b>", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#aaa"},
            "bar": {"color": color},
            "bgcolor": "rgba(30,30,40,0.8)",
            "steps": [
                {"range": [0, 33], "color": "rgba(46,204,113,0.3)"},
                {"range": [33, 66], "color": "rgba(241,196,15,0.3)"},
                {"range": [66, 100], "color": "rgba(231,76,60,0.3)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": risk_score * 100,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )
    return fig


def _risk_color(score: float) -> str:
    if score >= 0.66:
        return "#e74c3c"
    elif score >= 0.33:
        return "#f39c12"
    return "#2ecc71"


def risk_label(score: float) -> str:
    """Return a review-priority label. These are triage signals only — not diagnostic classifications."""
    if score >= 0.80:
        return "HIGH PRIORITY — Warrants Immediate Review"
    elif score >= 0.66:
        return "ELEVATED PRIORITY — Review Recommended"
    elif score >= 0.33:
        return "UNCERTAIN — Inconclusive Signal"
    elif score >= 0.20:
        return "LOW PRIORITY — Lower Review Signal"
    return "MINIMAL SIGNAL"


# ── CSV export ────────────────────────────────────────────────────────────────

def results_to_csv(results: list) -> bytes:
    """Convert a list of result dicts to CSV bytes."""
    df = pd.DataFrame(results)
    return df.to_csv(index=False).encode("utf-8")
