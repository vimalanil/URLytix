"""
Batch Enrichment: Add WHOIS / DNS / SSL / Live-Web / HTML-DOM Features
=========================================================================
Runs domain_web_features.py's extractor across your PhiUSIIL dataset
(or any URL list) with:
  - Multithreading (I/O-bound lookups benefit heavily from concurrency)
  - Checkpointing every N rows -> safe to stop/resume, survives crashes
  - Rate-limit-friendly pacing (randomized delay + small thread pool)
  - Stratified sampling helper, since running all 235K rows is NOT
    realistic against public WHOIS servers (you will get IP-banned)

USAGE
-----
Step 1 - take a manageable sample (recommended, not the full 235K):
    python batch_enrich_features.py --input PhiUSIIL_Phishing_URL_Dataset.csv \
        --sample 5000 --output enriched_sample.csv

Step 2 - resume if it got interrupted (reads existing output, skips done rows):
    python batch_enrich_features.py --input PhiUSIIL_Phishing_URL_Dataset.csv \
        --sample 5000 --output enriched_sample.csv --resume

Step 3 - once you trust it, scale up gradually (e.g. 20000), never jump to 235K.
"""

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from domain_web_features import extract_all_domain_web_features


def stratified_sample(df: pd.DataFrame, label_col: str, n: int, random_state: int = 42) -> pd.DataFrame:
    """Sample n rows preserving the original class balance."""
    frac = n / len(df)
    # groupby(...).sample(frac=...) keeps all columns (incl. the group column)
    # unlike groupby(...).apply(lambda x: x.sample(...)), which can silently
    # drop the grouping column on some pandas versions.
    return (
        df.groupby(label_col, group_keys=False)
        .sample(frac=frac, random_state=random_state)
        .reset_index(drop=True)
    )


def load_existing_progress(output_path: str) -> pd.DataFrame:
    if os.path.exists(output_path):
        print(f"Resuming from existing file: {output_path}")
        return pd.read_csv(output_path)
    return pd.DataFrame()


def enrich_one(row_idx: int, url: str, fetch_web_content: bool) -> dict:
    """Wrapper so we can track which row an extraction result belongs to."""
    try:
        feats = extract_all_domain_web_features(url, fetch_web_content=fetch_web_content)
    except Exception as e:
        feats = {"error": str(e)}
    feats["_row_idx"] = row_idx
    feats["URL"] = url
    return feats


def run_batch(
    df: pd.DataFrame,
    url_col: str,
    output_path: str,
    max_workers: int = 8,
    fetch_web_content: bool = True,
    checkpoint_every: int = 100,
    min_delay: float = 0.1,
    max_delay: float = 0.5,
):
    done_df = load_existing_progress(output_path)
    done_indices = set(done_df["_row_idx"]) if not done_df.empty else set()

    remaining = df[~df.index.isin(done_indices)]
    print(f"Total rows: {len(df)} | Already done: {len(done_indices)} | Remaining: {len(remaining)}")

    if remaining.empty:
        print("Nothing left to process.")
        return done_df

    results = list(done_df.to_dict("records")) if not done_df.empty else []
    processed_since_checkpoint = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, row in remaining.iterrows():
            # stagger submissions slightly to be polite to WHOIS/DNS/web servers
            time.sleep(random.uniform(min_delay, max_delay) / max_workers)
            futures[executor.submit(enrich_one, idx, row[url_col], fetch_web_content)] = idx

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            processed_since_checkpoint += 1

            if processed_since_checkpoint >= checkpoint_every:
                pd.DataFrame(results).to_csv(output_path, index=False)
                print(f"  checkpoint saved: {len(results)}/{len(df)} rows done")
                processed_since_checkpoint = 0

    # final save
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Done. Final output saved to {output_path} ({len(results)} rows)")
    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to PhiUSIIL CSV (or any CSV with a URL column)")
    parser.add_argument("--url_col", default="URL")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--sample", type=int, default=0, help="Stratified sample size. 0 = use full input as-is (NOT recommended for 235K rows)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--no_web_content", action="store_true", help="Skip live page fetch, only WHOIS/DNS/SSL")
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.sample > 0:
        df = stratified_sample(df, args.label_col, args.sample)
        print(f"Stratified sample taken: {len(df)} rows")

    run_batch(
        df=df,
        url_col=args.url_col,
        output_path=args.output,
        max_workers=args.max_workers,
        fetch_web_content=not args.no_web_content,
        checkpoint_every=args.checkpoint_every,
    )
