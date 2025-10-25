import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path

from etl.pipeline import build_final_panel  # import your functions and return df_final

def main(out_dir: str, stamp: str | None = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build panel (your pipeline returns the tidy modeling table)
    df_final = build_final_panel()

    # Build a filename with the “data month” (choose whatever makes sense for you)
    if stamp is None:
        stamp = datetime.utcnow().strftime("%Y-%m")  # e.g., 2025-10
    fname = f"panel_{stamp}.parquet"
    fpath = out / fname

    # Fast, compressed Parquet
    df_final.to_parquet(fpath, index=False, engine="pyarrow", compression="zstd")

    # Update latest pointer (simple copy)
    (out / "latest.parquet").unlink(missing_ok=True)
    df_final.to_parquet(out / "latest.parquet", index=False, engine="pyarrow", compression="zstd")

    # Optional small manifest for the app
    pd.Series({"latest": fname}).to_json(out / "manifest.json")
    print(f"Wrote {fpath} and latest.parquet")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--stamp", default=None, help="YYYY-MM for the panel filename")
    args = ap.parse_args()
    main(args.out_dir, args.stamp)
