"""
Enrichment Summary Report
==========================
Run this after batch_enrich_features.py finishes (or partway through,
on a checkpoint) to see:
  - How many URLs are still live vs dead/unreachable
  - WHOIS / DNS / SSL / web-fetch success rates, overall and by label
  - Quick sanity distributions for the new features, split by phishing
    vs legitimate, so you can eyeball whether they're separating classes
    before you bother merging/retraining

USAGE
-----
    python enrichment_summary.py --enriched enriched_5k.csv \
        --original PhiUSIIL_Phishing_URL_Dataset.csv
"""

import argparse
import pandas as pd


def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def summarize(enriched_path: str, original_path: str = None):
    df = pd.read_csv(enriched_path)
    print(f"Loaded {len(df)} enriched rows from {enriched_path}")

    if "error" in df.columns:
        hard_errors = df["error"].notna().sum()
        if hard_errors > 0:
            print(f"WARNING: {hard_errors} rows failed with an unhandled exception "
                  f"(see the 'error' column) - these have no usable features.")
            df = df[df["error"].isna()].copy()

    # Merge label back in from the original file if not already present
    if "label" not in df.columns:
        if original_path is None:
            print("No 'label' column found and no --original file given; "
                  "skipping label-split analysis.")
            label_available = False
        else:
            orig = pd.read_csv(original_path)[["URL", "label"]]
            df = df.merge(orig, on="URL", how="left")
            label_available = df["label"].notna().any()
    else:
        label_available = True

    # ------------------------------------------------------------
    # 1. Lookup success/failure rates
    # ------------------------------------------------------------
    print_section("1. LOOKUP SUCCESS / FAILURE RATES")
    failure_cols = ["WhoisLookupFailed", "DNSLookupFailed", "FetchFailed"]
    for col in failure_cols:
        if col in df.columns:
            rate = df[col].mean() * 100
            print(f"{col:22s}: {rate:5.1f}% of rows failed  "
                  f"({int(df[col].sum())}/{len(df)})")

    if "ResponseStatusCode" in df.columns:
        print("\nResponse status code distribution (web fetch):")
        print(df["ResponseStatusCode"].value_counts().sort_index().to_string())

    # ------------------------------------------------------------
    # 2. Failure rates split by label (phishing vs legitimate)
    # ------------------------------------------------------------
    if label_available:
        print_section("2. FAILURE RATES BY LABEL (0=phishing, 1=legitimate)")
        for col in failure_cols:
            if col in df.columns:
                print(f"\n{col}:")
                print(df.groupby("label")[col].mean().apply(lambda x: f"{x*100:.1f}%").to_string())

    # ------------------------------------------------------------
    # 3. Live vs dead breakdown
    # ------------------------------------------------------------
    print_section("3. LIVE VS DEAD URL BREAKDOWN")
    if "FetchFailed" in df.columns and "ResponseStatusCode" in df.columns:
        alive = df[(df["FetchFailed"] == 0) & (df["ResponseStatusCode"] == 200)]
        dead = df[(df["FetchFailed"] == 1) | (df["ResponseStatusCode"] != 200)]
        print(f"Live (200 OK):     {len(alive)} rows ({len(alive)/len(df)*100:.1f}%)")
        print(f"Dead/unreachable:  {len(dead)} rows ({len(dead)/len(df)*100:.1f}%)")

        if label_available:
            print("\nLive rate by label:")
            df["_is_live"] = ((df["FetchFailed"] == 0) & (df["ResponseStatusCode"] == 200)).astype(int)
            print(df.groupby("label")["_is_live"].mean().apply(lambda x: f"{x*100:.1f}% live").to_string())

    # ------------------------------------------------------------
    # 4. Feature distributions by label (quick sanity check)
    # ------------------------------------------------------------
    if label_available:
        print_section("4. FEATURE MEANS BY LABEL (sanity check - do these separate classes?)")
        print("Note: each group below is averaged ONLY over rows where that "
              "lookup actually succeeded, so failure-default values (e.g. "
              "TitleDomainMismatch defaulting to 1 when the fetch failed) "
              "don't contaminate the comparison.\n")

        groups = {
            "WHOIS-based (rows where WhoisLookupFailed == 0)": (
                "WhoisLookupFailed", ["DomainAgeDays", "DomainExpiryDays", "WhoisPrivacyProtected"]
            ),
            "DNS-based (rows where DNSLookupFailed == 0)": (
                "DNSLookupFailed", ["HasMXRecord", "HasSPFRecord", "NoOfNSRecords", "DNSDomainNotFound"]
            ),
            "SSL-based (rows where HasValidSSL == 1)": (
                None, ["SSLCertAgeDays", "SSLDaysToExpiry", "SSLIssuerIsKnownCA"]
            ),
            "Web-content-based (rows where FetchFailed == 0)": (
                "FetchFailed", ["HasPasswordForm", "FormActionExternalDomain",
                                 "ExternalResourceRatio", "TitleDomainMismatch",
                                 "NoOfRedirects", "FinalURLDiffersFromInput"]
            ),
        }

        for group_name, (fail_col, cols) in groups.items():
            present = [c for c in cols if c in df.columns]
            if not present:
                continue
            if fail_col and fail_col in df.columns:
                subset = df[df[fail_col] == 0]
            elif fail_col is None and "HasValidSSL" in df.columns:
                subset = df[df["HasValidSSL"] == 1]
            else:
                subset = df
            print(f"\n--- {group_name} | n={len(subset)} ---")
            if len(subset) == 0:
                print("  (no successful rows in this group)")
                continue
            means = subset.groupby("label")[present].mean().T
            print(means.round(3).to_string())

    print_section("DONE")
    print(f"Enriched, usable rows: {len(df)}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched", required=True, help="Output CSV from batch_enrich_features.py")
    parser.add_argument("--original", default=None,
                         help="Original PhiUSIIL CSV, needed if the enriched file doesn't already have a 'label' column")
    args = parser.parse_args()

    summarize(args.enriched, args.original)
