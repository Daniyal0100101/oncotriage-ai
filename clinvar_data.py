"""
clinvar_data.py — OncoTriage AI

ClinVar NCBI E-utilities + BRCA Exchange API wrappers.
"""

import time
import requests
import pandas as pd
from typing import Optional, Dict, List

# ── NCBI E-utilities ──────────────────────────────────────────────────────────

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{NCBI_BASE}/esearch.fcgi"
ESUMMARY_URL = f"{NCBI_BASE}/esummary.fcgi"
EFETCH_URL = f"{NCBI_BASE}/efetch.fcgi"

# BRCA Exchange
BRCA_EXCHANGE_BASE = "https://brcaexchange.org/backend/data"

# Rate-limit: NCBI allows 3 req/s without key, 10/s with key
_NCBI_DELAY = 0.35

SIGNIFICANCE_MAP = {
    "Pathogenic": 1,
    "Likely pathogenic": 1,
    "Benign": 0,
    "Likely benign": 0,
}


# ── ClinVar helpers ───────────────────────────────────────────────────────────

def search_clinvar_brca(gene: str = "BRCA1",
                         significance: str = "pathogenic",
                         max_results: int = 500,
                         ncbi_api_key: Optional[str] = None) -> List[str]:
    """
    Search ClinVar for BRCA1 or BRCA2 variants with a given clinical significance.
    Returns a list of ClinVar UIDs.
    """
    query = f"{gene}[gene] AND {significance}[clinical significance]"
    params = {
        "db": "clinvar",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    if ncbi_api_key:
        params["api_key"] = ncbi_api_key

    try:
        resp = requests.get(ESEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        return ids
    except Exception as e:
        print(f"ClinVar esearch error: {e}")
        return []


def fetch_clinvar_summaries(uids: List[str],
                             ncbi_api_key: Optional[str] = None,
                             batch_size: int = 100) -> List[Dict]:
    """
    Fetch esummary records for a list of ClinVar UIDs.
    Returns list of variant summary dicts.
    """
    if not uids:
        return []

    records = []
    for i in range(0, len(uids), batch_size):
        batch = uids[i:i + batch_size]
        params = {
            "db": "clinvar",
            "id": ",".join(batch),
            "retmode": "json",
        }
        if ncbi_api_key:
            params["api_key"] = ncbi_api_key

        try:
            resp = requests.get(ESUMMARY_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("result", {})
            for uid in batch:
                if uid in result:
                    records.append(result[uid])
        except Exception as e:
            print(f"ClinVar esummary error (batch {i//batch_size}): {e}")

        time.sleep(_NCBI_DELAY)

    return records


def parse_clinvar_records(records: List[Dict]) -> pd.DataFrame:
    """
    Parse ClinVar esummary records into a clean DataFrame.
    """
    rows = []
    for rec in records:
        # Extract clinical significance
        clin_sig_list = rec.get("clinical_significance", {})
        if isinstance(clin_sig_list, dict):
            sig = clin_sig_list.get("description", "")
        elif isinstance(clin_sig_list, str):
            sig = clin_sig_list
        else:
            sig = ""

        # Extract variant name (HGVS)
        title = rec.get("title", "")
        obj_type = rec.get("obj_type", "")
        gene_sort = rec.get("gene_sort", "")

        # Extract protein change
        variation_set = rec.get("variation_set", [{}])
        hgvs_names = []
        if variation_set:
            for v in variation_set if isinstance(variation_set, list) else [variation_set]:
                measures = v.get("variation_xrefs", [])

        rows.append({
            "uid": rec.get("uid", ""),
            "title": title,
            "gene": gene_sort,
            "obj_type": obj_type,
            "clinical_significance": sig,
            "label": SIGNIFICANCE_MAP.get(sig, -1),
            "review_status": rec.get("reviewstatus", ""),
        })

    df = pd.DataFrame(rows)
    # Keep only pathogenic/benign (label != -1)
    df = df[df["label"] != -1].reset_index(drop=True)
    return df


def download_brca_clinvar_dataset(max_per_class: int = 500,
                                   ncbi_api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Download labeled ClinVar BRCA1+BRCA2 variants.
    Returns DataFrame with columns: uid, title, gene, clinical_significance, label.
    """
    all_frames = []

    for gene in ["BRCA1", "BRCA2"]:
        for sig, label in [("pathogenic", 1), ("benign", 0)]:
            print(f"Fetching {gene} {sig} variants from ClinVar...")
            uids = search_clinvar_brca(gene, sig, max_per_class, ncbi_api_key)
            print(f"  Found {len(uids)} UIDs")
            records = fetch_clinvar_summaries(uids, ncbi_api_key)
            df = parse_clinvar_records(records)
            df["gene"] = gene
            df["label"] = label
            all_frames.append(df)
            time.sleep(_NCBI_DELAY)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["uid"])


# ── BRCA Exchange API ─────────────────────────────────────────────────────────

def query_brca_exchange(search_term: str, page_size: int = 100) -> Dict:
    """
    Query BRCA Exchange live database for a variant or gene.
    Returns raw API response dict.
    """
    params = {
        "format": "json",
        "search_term": search_term,
        "page_size": page_size,
        "order_by": "Gene_symbol",
    }
    try:
        resp = requests.get(BRCA_EXCHANGE_BASE, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "count": 0, "results": []}


def parse_brca_exchange_results(api_resp: Dict) -> pd.DataFrame:
    """
    Parse BRCA Exchange API response into a clean DataFrame.
    """
    results = api_resp.get("results", [])
    if not results:
        return pd.DataFrame()

    rows = []
    for item in results:
        hgvs = item.get("HGVS_cDNA", "") or item.get("Genomic_Coordinate_hg38", "")
        sig = item.get("Pathogenicity_all", "") or item.get("Clinical_significance_ENIGMA", "")
        rows.append({
            "hgvs": hgvs,
            "gene": item.get("Gene_symbol", ""),
            "pathogenicity": sig,
            "source": "BRCA Exchange",
            "url": item.get("url", ""),
        })

    return pd.DataFrame(rows)


def lookup_variant_brca_exchange(hgvs_notation: str) -> Optional[Dict]:
    """
    Look up a specific variant in BRCA Exchange.
    Returns the best match or None.
    """
    resp = query_brca_exchange(hgvs_notation, page_size=10)
    results = resp.get("results", [])
    if results:
        return results[0]
    return None


# ── BRCA reference sequences (short demo contexts) ────────────────────────────

# Short representative BRCA1 reference contexts for demo use.
# In production, fetch from Ensembl REST API or UCSC.
BRCA1_DEMO_CONTEXTS = {
    # Exon 11 known pathogenic hotspot region (synthetic 200bp context)
    "exon11": "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGCTGAAACTTCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGAT",
    # Exon 2 (includes BRCA1 start)
    "exon2": "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACC",
}

BRCA2_DEMO_CONTEXTS = {
    # Exon 11 (longest exon, many variants)
    "exon11": "CAGTTTGTTTCAGAATCTAAAAAAATTGAAATTTTCATTGAAGACAAAGTTTTGAATGAATTTGAAAAGCAAGTTTTGAAAGCAGAAGAATTTGAAAGCAGATGAATTTGAAAGCAGAAGAATTTGAAAGCAGATGAATTTGAAAGCAGAAGAATTTGAAAGCAGATGAATTTGAAAGCAGAAGAATTTGAAAGCAGAT",
}


def get_demo_sequence(gene: str = "BRCA1", region: str = "exon11") -> str:
    """Return a demo reference sequence for testing."""
    if gene.upper() == "BRCA1":
        return BRCA1_DEMO_CONTEXTS.get(region, BRCA1_DEMO_CONTEXTS["exon11"])
    elif gene.upper() == "BRCA2":
        return BRCA2_DEMO_CONTEXTS.get(region, BRCA2_DEMO_CONTEXTS["exon11"])
    return BRCA1_DEMO_CONTEXTS["exon11"]


# ── Variant status summary ────────────────────────────────────────────────────

def summarize_known_variant_status(hgvs: str) -> Dict:
    """
    Look up a variant in BRCA Exchange and return a human-readable status summary.
    """
    result = lookup_variant_brca_exchange(hgvs)
    if not result:
        return {
            "found": False,
            "message": "Not found in BRCA Exchange database.",
            "pathogenicity": "Unknown",
            "source": "BRCA Exchange",
        }
    return {
        "found": True,
        "message": f"Known variant in BRCA Exchange",
        "pathogenicity": result.get("Pathogenicity_all", "Unknown"),
        "gene": result.get("Gene_symbol", ""),
        "hgvs": result.get("HGVS_cDNA", hgvs),
        "source": "BRCA Exchange",
    }
