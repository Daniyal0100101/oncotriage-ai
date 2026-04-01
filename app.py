"""
app.py — OncoTriage AI
Research prototype Streamlit application powered by NVIDIA NIM Evo 2.
Research use only. Outputs are not clinically validated.
"""

import os
from dotenv import load_dotenv
import io
import time
import traceback
import pandas as pd
import streamlit as st

from model import (
    Evo2Client, BRCAScorer, build_synthetic_training_data,
    InferenceUnavailableError, _demo_score_result,
)
from utils import (
    parse_variant,
    validate_dna_sequence,
    apply_variant_to_sequence,
    annotate_variant,
    make_impact_bar_chart,
    make_genome_browser_snippet,
    make_risk_gauge,
    risk_label,
    results_to_csv,
)
from clinvar_data import get_demo_sequence, summarize_known_variant_status

# ── Page configuration ────────────────────────────────────────────────────────

load_dotenv()

st.set_page_config(
    page_title="OncoTriage AI · Evo 2 Variant Scoring",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Daniyal0100101/oncotriage-ai",
        "Report a bug": "https://github.com/Daniyal0100101/oncotriage-ai/issues",
        "About": "OncoTriage AI — Research-use hereditary cancer variant review copilot. Not for clinical use.",
    },
)

# ── Custom CSS / Dark theme ───────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0a0a14 0%, #0d1117 40%, #0a0f1e 100%);
    color: #e0e6f0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
    border-right: 1px solid rgba(99,102,241,0.3);
  }

  /* Hero title */
  .hero-title {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.2rem;
  }
  .hero-sub {
    font-size: 1.05rem; color: #94a3b8; margin-bottom: 1.5rem; letter-spacing: 0.02em;
  }

  /* Disclaimer */
  .disclaimer-box {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border: 1.5px solid rgba(239,68,68,0.6);
    border-radius: 12px; padding: 1rem 1.4rem;
    margin: 1rem 0 1.5rem 0;
    color: #fca5a5; font-size: 0.92rem;
  }
  .disclaimer-box b { color: #f87171; font-size: 1rem; }

  /* Cards */
  .score-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px; padding: 1.4rem;
    backdrop-filter: blur(10px);
  }

  /* Status badges */
  .badge-high   { background:#ef4444; color:#fff; border-radius:6px; padding:3px 10px; font-weight:600; font-size:0.85rem; }
  .badge-medium { background:#f59e0b; color:#fff; border-radius:6px; padding:3px 10px; font-weight:600; font-size:0.85rem; }
  .badge-low    { background:#10b981; color:#fff; border-radius:6px; padding:3px 10px; font-weight:600; font-size:0.85rem; }

  /* Sequence textarea */
  textarea { font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important; }

  /* Divider */
  hr { border-color: rgba(99,102,241,0.2) !important; }

  /* Metrics */
  [data-testid="stMetricValue"] { color: #a5b4fc !important; font-weight: 700; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; font-size: 1rem; padding: 0.6rem 1.6rem;
    transition: all 0.25s ease;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    transform: translateY(-1px); box-shadow: 0 4px 20px rgba(99,102,241,0.4);
  }

  /* Expanders */
  .streamlit-expanderHeader {
    background: rgba(15,23,42,0.6) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "results": [],
        "last_score": None,
        "scorer": None,
        "trained": False,
        "api_key": os.environ.get("NVIDIA_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧬 OncoTriage AI")
    st.markdown("*Powered by NVIDIA NIM Evo 2 (40B)*")
    st.divider()

    api_key = st.text_input(
        "🔑 NVIDIA NIM API Key",
        type="password",
        placeholder="nvapi-...",
        value=st.session_state.api_key,
        help="Get a free key at https://build.nvidia.com/arc/evo2-40b",
    )
    if api_key:
        st.session_state.api_key = api_key
        st.success("✅ API key set", icon="🔑")

    st.divider()
    st.markdown("**Model Settings**")
    model_size = st.selectbox("Evo 2 Model", ["evo2-40b", "evo2-7b"],
                               help="40B is more accurate; 7B is faster.")
    context_window = st.slider(
        "Context window (bp)", min_value=128, max_value=4096,
        value=512, step=128,
        help="Bases around the variant used for scoring.",
    )
    temperature = st.slider("Generation temperature", 0.1, 1.5, 0.7, 0.05)
    num_gen_tokens = st.number_input("Generated tokens (bonus)", 50, 500, 100, 10)

    st.divider()
    st.markdown("**About**")
    st.caption(
        "OncoTriage AI uses Evo 2's /generate endpoint to derive per-sequence "
        "log-likelihood proxies for ref vs alt alleles. A lightweight RandomForest "
        "ranking head maps these features to a variant review priority score."
    )
    st.caption(
        "📄 [Evo 2 Nature paper](https://doi.org/10.1038/s41586-026-10176-5) · "
        "[ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) · "
        "[BRCA Exchange](https://brcaexchange.org)"
    )
    st.divider()

    # Train ML head button
    if st.button("🤖 Pre-train ML Head (offline)", use_container_width=True):
        with st.spinner("Training RandomForest on synthetic ClinVar-like data…"):
            scorer = BRCAScorer()
            syn_data = build_synthetic_training_data(300, 300)
            metrics = scorer.train_on_clinvar(syn_data)
            st.session_state.scorer = scorer
            st.session_state.trained = True
            st.success(
                f"✅ Trained! CV-AUC: {metrics['auc_cv_mean']:.3f} ± {metrics['auc_cv_std']:.3f}",
                icon="🎯",
            )


# ── Main layout ───────────────────────────────────────────────────────────────

st.markdown('<h1 class="hero-title">OncoTriage AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Real-time hereditary cancer variant review copilot '
    'powered by the Evo 2 (40B) genomic foundation model.</p>',
    unsafe_allow_html=True,
)

if not st.session_state.api_key:
    st.warning("🔄 **DEMO MODE ACTIVE**: No NVIDIA API Key found. The application is running using offline simulated/mock inference data to demonstrate functionality.")

# ── ⚠ Disclaimer ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="disclaimer-box">
  <b>⚠️ RESEARCH/EDUCATIONAL USE ONLY — NOT FOR CLINICAL DIAGNOSIS</b><br>
  This tool is intended solely for research and educational purposes. 
  Results are <u>not validated for clinical decision-making</u> and must 
  <b>never be used as a basis for medical decisions or treatment</b>. 
  Always consult qualified medical professionals or a certified clinical geneticist for interpretation.
  Pathogenicity predictions may be inaccurate or incomplete.
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_single, tab_batch, tab_generate, tab_data = st.tabs([
    "🔬 Single Variant",
    "📋 Batch Scoring",
    "🧬 Generate Variants",
    "📊 ClinVar Explorer",
])

# =============================================================================
# Tab 1: Single variant scoring
# =============================================================================

with tab_single:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown("### Input")

        # Gene / demo sequence selector
        gene_choice = st.selectbox("Reference gene", ["BRCA1", "BRCA2", "Custom"])
        if gene_choice != "Custom":
            demo_seq = get_demo_sequence(gene_choice, "exon11")
            seq_help = f"Pre-loaded {gene_choice} exon 11 demo context. Paste your own sequence to override."
        else:
            demo_seq = ""
            seq_help = "Paste a raw DNA sequence (A/C/T/G only)."

        sequence = st.text_area(
            "📄 Reference DNA Sequence",
            value=demo_seq,
            height=160,
            placeholder="ATGGATTTATCTGCTCTTCGCGTTGAAG...",
            help=seq_help,
        )

        variant_input = st.text_input(
            "🔀 Variant notation",
            placeholder="c.5123C>A  or  185 A>G  or  g.41246481C>T",
            help="Supports HGVS cDNA (c.NNN X>Y), genomic (g.NNN X>Y), or positional (pos ref>alt).",
        )

        score_btn = st.button("⚡ Score with Evo 2", use_container_width=True)
        if not st.session_state.api_key:
            st.caption("🔑 App is currently in DEMO MODE.")

        # BRCA Exchange lookup
        if variant_input:
            with st.expander("🔍 BRCA Exchange Database Lookup", expanded=False):
                with st.spinner("Querying BRCA Exchange…"):
                    brca_status = summarize_known_variant_status(variant_input)
                if brca_status["found"]:
                    st.success(f"**Found:** {brca_status['hgvs']}")
                    st.write(f"**External DB classification:** {brca_status['pathogenicity']}")
                    st.caption("Source: BRCA Exchange (brcaexchange.org)")
                else:
                    st.info(
                        "ℹ️ **Not found in BRCA Exchange** — this variant is not in the "
                        "database (may be novel or unlisted). This is a **database lookup "
                        "result only** and does not indicate any Evo 2 scoring failure."
                    )

    with col_right:
        st.markdown("### Results")

        if score_btn:
            # ── Validate inputs ───────────────────────────────────────────────
            valid_seq, seq_err = validate_dna_sequence(sequence)
            if not valid_seq:
                st.error(f"Sequence error: {seq_err}")
                st.stop()

            if not variant_input.strip():
                st.error("Please enter a variant notation.")
                st.stop()

            parsed = parse_variant(variant_input.strip())
            if parsed is None:
                st.error(
                    f"Could not parse variant: **{variant_input}**\n\n"
                    "Use format: `c.5123C>A`, `g.41246481C>T`, or `185 C>A`."
                )
                st.stop()

            # Prevent identical bases
            if parsed["ref"] == parsed["alt"]:
                st.error("Reference and alternate alleles cannot be the same.")
                st.stop()

            # ── Score ─────────────────────────────────────────────────────────
            try:
                with st.spinner("🧠 Preparing sequences…"):
                    ref_ctx, alt_ctx = apply_variant_to_sequence(
                        sequence, parsed, context_window
                    )
                    client = Evo2Client(st.session_state.api_key)
                    # Patch model if user chose 7b
                    if model_size == "evo2-7b":
                        import model as _m
                        _m.EVO2_MODEL = "evo2-7b"
                        _m.FORWARD_ENDPOINT = f"{_m.NIM_BASE_URL}/biology/arc/evo2-7b/forward"
                        _m.GENERATE_ENDPOINT = f"{_m.NIM_BASE_URL}/biology/arc/evo2-7b/generate"
                        client = Evo2Client(st.session_state.api_key)

                is_demo = not bool(st.session_state.api_key)
                fallback_occurred = False

                with st.spinner("🧠 Running Evo 2 variant scoring…"):
                    try:
                        score_result = client.score_variant(ref_ctx, alt_ctx)
                        is_demo = is_demo or score_result.get("demo_mode", False)
                    except InferenceUnavailableError as api_err:
                        st.warning(
                            "⚠️ **Live Evo 2 inference is temporarily unavailable** — "
                            "the NVIDIA API returned an internal error. "
                            "Results below are **demo-mode estimates only** and do not "
                            "reflect real model outputs.\n\n"
                            f"*Technical detail:* `{api_err}`"
                        )
                        score_result = _demo_score_result()
                        is_demo = True
                        fallback_occurred = True

                if is_demo and st.session_state.api_key:
                    st.info(
                        "🔬 **Demo-mode estimates** — values generated from a calibrated "
                        "random distribution; **not** live Evo 2 predictions."
                    )

                delta_ll = score_result["delta_ll"]
                annotation = annotate_variant(parsed, ref_ctx)

                scorer = st.session_state.scorer or BRCAScorer()
                risk = scorer.predict_risk(
                    delta_ll, score_result["ll_ref"], score_result["ll_alt"],
                    ref_ctx, alt_ctx, annotation,
                )

                # ── Store result ──────────────────────────────────────────────
                result_record = {
                    "variant": variant_input,
                    "gene": gene_choice,
                    "ref_allele": parsed["ref"],
                    "alt_allele": parsed["alt"],
                    "position": parsed["position"],
                    "ll_ref": round(score_result["ll_ref"], 5),
                    "ll_alt": round(score_result["ll_alt"], 5),
                    "delta_ll": round(delta_ll, 5),
                    "risk_score": round(risk, 4),
                    "classification": risk_label(risk),
                    "annotation": annotation,
                    "elapsed_ms": score_result["elapsed_ms_total"],
                    "inference_mode": "DEMO" if is_demo else "LIVE",
                    "scoring_path": "demo" if is_demo else "generate-proxy",
                    "model": model_size,
                    "fallback_occurred": fallback_occurred,
                }
                st.session_state.results.append(result_record)
                st.session_state.last_score = result_record

            except (InferenceUnavailableError, RuntimeError) as e:
                st.error(str(e))
                st.stop()
            except Exception:
                st.error(f"Unexpected error:\n\n```\n{traceback.format_exc()}\n```")
                st.stop()

        # ── Display latest results ────────────────────────────────────────────
        if st.session_state.last_score:
            r = st.session_state.last_score
            st.plotly_chart(make_risk_gauge(r["risk_score"]),
                            width="stretch", config={"displayModeBar": False})

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Δ Log-Likelihood", f"{r['delta_ll']:+.4f}",
                          help="Negative = damaging (alt less likely than ref under Evo 2).")
            with col_m2:
                st.metric("Review Priority", f"{r['risk_score']:.3f}")
            with col_m3:
                st.metric("Elapsed", f"{r['elapsed_ms']} ms")

            if r.get("inference_mode") == "DEMO":
                st.warning("🔬 These results are **demo-mode estimates** — not live Evo 2 predictions.")
            st.markdown(f"**Review signal:** {r['classification']}")
            st.markdown(f"**Annotation:** {r['annotation']}")
            st.caption(
                f"Inference mode: **{r.get('inference_mode', 'LIVE')}** · "
                f"Scoring path: {r.get('scoring_path', 'generate-proxy')} · "
                f"Model: {r.get('model', 'evo2-40b')}"
            )

    # ── Vizualisations below ──────────────────────────────────────────────────
    if st.session_state.last_score:
        r = st.session_state.last_score
        st.divider()
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.plotly_chart(
                make_impact_bar_chart(r["delta_ll"], r["risk_score"]),
                width="stretch", config={"displayModeBar": False},
            )
        with viz_col2:
            try:
                pos_int = int(str(r["position"]).replace("c.", "").split("+")[0].split("-")[0])
            except (ValueError, AttributeError):
                pos_int = 100
            parsed_v = parse_variant(r["variant"]) or {}
            display_seq = apply_variant_to_sequence(
                sequence if sequence else get_demo_sequence(gene_choice),
                parsed_v if parsed_v else {"position": pos_int, "ref": r["ref_allele"], "alt": r["alt_allele"], "type": "positional"},
                512,
            )[0]
            st.plotly_chart(
                make_genome_browser_snippet(
                    display_seq, pos_int,
                    r["ref_allele"], r["alt_allele"],
                ),
                width="stretch", config={"displayModeBar": False},
            )

        with st.expander("📋 Full Result Details", expanded=False):
            st.json(r)

        with st.expander("🔍 Result Provenance", expanded=False):
            _mode = r.get("inference_mode", "LIVE")
            _path = r.get("scoring_path", "generate-proxy")
            _mdl  = r.get("model", "evo2-40b")
            _live = "No" if _mode == "DEMO" else "Yes"
            _fall = "Yes" if r.get("fallback_occurred", False) else "No"
            st.markdown(
                f"| Field | Value |\n"
                f"|---|---|\n"
                f"| Model | `{_mdl}` |\n"
                f"| Inference path | `{_path}` |\n"
                f"| Mode | `{_mode}` |\n"
                f"| Live inference succeeded | `{_live}` |\n"
                f"| Demo fallback occurred | `{_fall}` |\n"
                f"| External lookup | See BRCA Exchange expander in input panel |"
            )
            st.caption("⚠️ Research-use variant review aid only — not a diagnostic tool.")


# =============================================================================
# Tab 2: Batch scoring
# =============================================================================

with tab_batch:
    st.markdown("### Batch Variant Scoring")
    st.markdown(
        "Upload a CSV with columns: `sequence`, `variant`, `gene` (optional). "
        "OncoTriage AI will score each row using Evo 2."
    )

    # Template download
    template_df = pd.DataFrame({
        "gene": ["BRCA1", "BRCA2"],
        "sequence": [
            get_demo_sequence("BRCA1")[:80],
            get_demo_sequence("BRCA2")[:80],
        ],
        "variant": ["c.185G>A", "c.5946G>T"],
    })
    st.download_button(
        "⬇ Download CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="oncotriage_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload variant CSV", type=["csv"])

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        required_cols = {"sequence", "variant"}
        if not required_cols.issubset(batch_df.columns):
            st.error(f"CSV must contain columns: {required_cols}. Found: {set(batch_df.columns)}")
        else:
            if st.button("🚀 Run Batch Scoring", use_container_width=True):
                client = Evo2Client(st.session_state.api_key)
                scorer = st.session_state.scorer or BRCAScorer()
                progress = st.progress(0)
                status_txt = st.empty()
                batch_results = []
                n = len(batch_df)

                for idx, row in batch_df.iterrows():
                    status_txt.text(f"Scoring variant {idx+1}/{n}: {row['variant']}")
                    try:
                        parsed = parse_variant(str(row["variant"]))
                        if parsed is None:
                            raise ValueError(f"Cannot parse variant: {row['variant']}")
                        ref_ctx, alt_ctx = apply_variant_to_sequence(
                            str(row["sequence"]), parsed, context_window
                        )
                        sr = client.score_variant(ref_ctx, alt_ctx)
                        ann = annotate_variant(parsed)
                        risk = scorer.predict_risk(
                            sr["delta_ll"], sr["ll_ref"], sr["ll_alt"],
                            ref_ctx, alt_ctx, ann,
                        )
                        batch_results.append({
                            "variant": row["variant"],
                            "gene": row.get("gene", ""),
                            "delta_ll": round(sr["delta_ll"], 5),
                            "risk_score": round(risk, 4),
                            "classification": risk_label(risk),
                            "annotation": ann,
                            "elapsed_ms": sr["elapsed_ms_total"],
                            "error": "",
                        })
                    except Exception as e:
                        batch_results.append({
                            "variant": row.get("variant", ""),
                            "gene": row.get("gene", ""),
                            "delta_ll": None, "risk_score": None,
                            "classification": "ERROR",
                            "annotation": str(e), "elapsed_ms": None,
                            "error": str(e),
                        })

                    progress.progress((idx + 1) / n)
                    time.sleep(0.1)  # be gentle with the API

                status_txt.success(f"✅ Scored {n} variants!")
                results_df = pd.DataFrame(batch_results)
                st.dataframe(results_df, use_container_width=True)
                st.download_button(
                    "⬇ Download Batch Results CSV",
                    data=results_to_csv(batch_results),
                    file_name="oncotriage_batch_results.csv",
                    mime="text/csv",
                )

    # Previous results table
    if st.session_state.results:
        st.divider()
        st.markdown("### Previous Single-Variant Results")
        prev_df = pd.DataFrame(st.session_state.results)
        st.dataframe(prev_df, use_container_width=True)
        st.download_button(
            "⬇ Download Results CSV",
            data=results_to_csv(st.session_state.results),
            file_name="oncotriage_results.csv",
            mime="text/csv",
        )


# =============================================================================
# Tab 3: Generate similar variants
# =============================================================================

with tab_generate:
    st.markdown("### 🧬 Generate Sequence Continuations via Evo 2")
    st.markdown(
        "Use Evo 2's generative capability to create synthetic DNA continuations "
        "from a known benign context. Useful for exploring sequence space around "
        "neutral variants."
    )
    st.info(
        "ℹ️ This feature calls the Evo 2 **generate** endpoint and does NOT guarantee "
        "that generated sequences are biologically valid or clinically benign. "
        "Always validate with wet-lab experiments."
    )

    gen_col1, gen_col2 = st.columns([1, 1], gap="large")
    with gen_col1:
        gen_gene = st.selectbox("Seed gene context", ["BRCA1", "BRCA2", "Custom"], key="gen_gene")
        if gen_gene != "Custom":
            gen_seq_default = get_demo_sequence(gen_gene)
        else:
            gen_seq_default = ""
        gen_sequence = st.text_area(
            "Seed sequence (prompt for generation)",
            value=gen_seq_default,
            height=140,
        )
        gen_tokens = st.slider("Tokens to generate", 50, 500, num_gen_tokens)
        gen_temp = st.slider("Temperature", 0.1, 1.5, temperature, key="gen_temp")
        gen_topk = st.selectbox("Top-k", [1, 2, 3, 4, 5], index=2)

        gen_btn = st.button("🔮 Generate", use_container_width=True)

    with gen_col2:
        st.markdown("**Generated Output**")
        if gen_btn:
            if not gen_sequence.strip():
                st.error("Please provide a seed sequence.")
                st.stop()
            valid_gen, gen_err = validate_dna_sequence(gen_sequence)
            if not valid_gen:
                st.error(f"Sequence error: {gen_err}")
                st.stop()
            try:
                with st.spinner("Generating DNA sequence with Evo 2…"):
                    client = Evo2Client(st.session_state.api_key)
                    gen_result = client.generate(
                        gen_sequence, gen_tokens, gen_temp, gen_topk,
                        enable_logits=False,
                    )
                generated_seq = gen_result.get("sequence", "")
                elapsed_gen = gen_result.get("elapsed_ms", 0)

                st.success(f"✅ Generated {len(generated_seq)} nucleotides in {elapsed_gen} ms")
                st.code(generated_seq, language="")

                full_seq = gen_sequence + generated_seq
                st.markdown(f"**Full sequence length:** {len(full_seq)} bp")
                st.download_button(
                    "⬇ Download generated sequence (FASTA)",
                    data=f">AI_Generated_{gen_gene}_context\n{full_seq}\n",
                    file_name="evo2_generated_variant.fasta",
                    mime="text/plain",
                )
            except RuntimeError as e:
                st.error(str(e))


# =============================================================================
# Tab 4: ClinVar Explorer
# =============================================================================

with tab_data:
    st.markdown("### 📊 ClinVar BRCA Data Explorer")
    st.markdown(
        "Pull BRCA1/BRCA2 variant summaries directly from ClinVar via NCBI "
        "E-utilities. Useful for research and for building custom training sets."
    )

    from clinvar_data import download_brca_clinvar_dataset, query_brca_exchange, parse_brca_exchange_results

    data_col1, data_col2 = st.columns([1, 1])
    with data_col1:
        fetch_gene = st.selectbox("Gene", ["BRCA1", "BRCA2", "Both"], key="fetch_gene")
        max_variants = st.slider("Max variants per class", 50, 500, 100, 50)
        ncbi_key_opt = st.text_input("NCBI API Key (optional, increases rate limit)",
                                      type="password", placeholder="optional")
        fetch_btn = st.button("📥 Fetch ClinVar Variants", use_container_width=True)

    with data_col2:
        st.markdown("**BRCA Exchange Quick Search**")
        brca_query = st.text_input("Search term (gene / variant / HGVS)",
                                    placeholder="BRCA1 c.185G>A", key="brca_query")
        brca_search_btn = st.button("🔍 Search BRCA Exchange", use_container_width=True)

    if fetch_btn:
        genes_to_fetch = (["BRCA1", "BRCA2"] if fetch_gene == "Both"
                          else [fetch_gene])
        all_dfs = []
        for g in genes_to_fetch:
            with st.spinner(f"Fetching {g} from ClinVar…"):
                from clinvar_data import (search_clinvar_brca,
                                          fetch_clinvar_summaries,
                                          parse_clinvar_records)
                rows = []
                for sig, lbl in [("pathogenic", 1), ("benign", 0)]:
                    uids = search_clinvar_brca(g, sig, max_variants,
                                               ncbi_key_opt or None)
                    recs = fetch_clinvar_summaries(uids, ncbi_key_opt or None)
                    df = parse_clinvar_records(recs)
                    df["gene"] = g
                    df["label"] = lbl
                    all_dfs.append(df)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            st.success(f"✅ Fetched {len(combined)} variants")
            st.dataframe(combined, use_container_width=True)
            st.download_button(
                "⬇ Download ClinVar Data CSV",
                data=combined.to_csv(index=False).encode("utf-8"),
                file_name="clinvar_brca_variants.csv",
                mime="text/csv",
            )

    if brca_search_btn and brca_query:
        with st.spinner("Querying BRCA Exchange…"):
            resp = query_brca_exchange(brca_query)
            brca_df = parse_brca_exchange_results(resp)
        if brca_df.empty:
            st.warning("No results from BRCA Exchange.")
        else:
            st.success(f"✅ {len(brca_df)} results from BRCA Exchange")
            st.dataframe(brca_df, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#475569; font-size:0.82rem; padding: 0.5rem 0 1rem;">
      OncoTriage AI · Evo 2 variant scoring · Not for clinical use ·
      <a href="https://doi.org/10.1038/s41586-026-10176-5" style="color:#6366f1;">Evo 2 paper</a> ·
      <a href="https://www.ncbi.nlm.nih.gov/clinvar/" style="color:#6366f1;">ClinVar</a> ·
      <a href="https://brcaexchange.org" style="color:#6366f1;">BRCA Exchange</a> ·
      Copyright © Daniyal Asif. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True,
)
