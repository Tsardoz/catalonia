"""
dun_rainfed_fraction.py
Compute per-comarca rainfed-area fraction for a single crop from a DUN parcel
CSV (e.g. olives.csv, almonds.csv from analisi.transparenciacatalunya.cat).

For each comarca:
  rainfed_area  = sum SUP_NETA_H over parcels with SIST_EXPL == 'S'
  irrigated_area = sum SUP_NETA_H over parcels with SIST_EXPL == 'R'
  rainfed_frac  = rainfed_area / (rainfed_area + irrigated_area)

Parcels below MIN_PARCEL_HA are dropped before aggregation. Areas are summed
across all years present in the file, then divided (treating the multi-year
file as one structural snapshot — DUN parcels are largely stable year-to-year).

Comarca name normalisation matches src/parse_yield.py so the output joins
cleanly to catalan_woody_yield_raw.csv.

CLI:
  python src/dun_rainfed_fraction.py --crop olive --input data/dun/olives.csv

Outputs:
  data/dun/comarca_{crop}_rainfed_fraction.csv  (full table, one row / comarca)
  data/dun/comarca_{crop}_whitelist.csv         (subset with rainfed_frac >= threshold)
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DUN_DIR = DATA_DIR / "dun"

# Match src/parse_yield.py to keep comarca names consistent across data sources
COMARCA_NORMALISE = {
    "La Selva":      "Selva",
    "la Selva":      "Selva",
    "Vall d'Aran":   "Aran",
    "Pla d 'Urgell": "Pla d'Urgell",
}

USECOLS = [
    "Campanya (CAMPANYA)",
    "Nom comarca (NOM_COM)",
    "Superfície neta en hectàrees (SUP_NETA_H)",
    "Sistema d\u2019explotaci\u00f3 (SIST_EXPL)",
]

RENAME = {
    "Campanya (CAMPANYA)":                              "year",
    "Nom comarca (NOM_COM)":                            "comarca",
    "Superf\u00edcie neta en hect\u00e0rees (SUP_NETA_H)": "sup_neta_h",
    "Sistema d\u2019explotaci\u00f3 (SIST_EXPL)":       "sist_expl",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop", required=True, help="Short crop name used in output filenames, e.g. 'olive', 'almond'")
    p.add_argument("--input", type=Path, required=True, help="Path to DUN parcel CSV for this crop")
    p.add_argument("--min-parcel-ha", type=float, default=0.5, help="Minimum SUP_NETA_H per parcel (default 0.5)")
    p.add_argument("--threshold", type=float, default=0.80, help="Rainfed-fraction whitelist cutoff (default 0.80)")
    p.add_argument("--chunksize", type=int, default=200_000, help="CSV read chunk size")
    return p.parse_args()


def aggregate(input_path: Path, min_parcel_ha: float, chunksize: int) -> pd.DataFrame:
    """Stream the DUN CSV and accumulate area sums per (comarca, sist_expl)."""
    parts = []
    reader = pd.read_csv(input_path, usecols=USECOLS, dtype=str, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        chunk = chunk.rename(columns=RENAME)
        chunk["sup_neta_h"] = pd.to_numeric(
            chunk["sup_neta_h"].str.replace(",", ".", regex=False), errors="coerce"
        )
        chunk = chunk.dropna(subset=["sup_neta_h", "comarca", "sist_expl"])
        chunk = chunk[chunk["sup_neta_h"] >= min_parcel_ha]
        chunk = chunk[chunk["sist_expl"].isin(["S", "R"])]
        chunk["comarca"] = chunk["comarca"].str.strip().replace(COMARCA_NORMALISE)
        parts.append(
            chunk.groupby(["comarca", "sist_expl"], as_index=False)
                 .agg(area_ha=("sup_neta_h", "sum"), n_parcels=("sup_neta_h", "size"))
        )

    long = pd.concat(parts, ignore_index=True)
    long = long.groupby(["comarca", "sist_expl"], as_index=False).sum()

    wide = long.pivot(index="comarca", columns="sist_expl", values=["area_ha", "n_parcels"]).fillna(0.0)
    wide.columns = [f"{m}_{s}" for m, s in wide.columns]
    wide = wide.rename(columns={
        "area_ha_S":   "rainfed_area_ha",
        "area_ha_R":   "irrigated_area_ha",
        "n_parcels_S": "rainfed_n_parcels",
        "n_parcels_R": "irrigated_n_parcels",
    }).reset_index()

    for c in ["rainfed_area_ha", "irrigated_area_ha", "rainfed_n_parcels", "irrigated_n_parcels"]:
        if c not in wide.columns:
            wide[c] = 0.0

    wide["total_area_ha"] = wide["rainfed_area_ha"] + wide["irrigated_area_ha"]
    wide["total_n_parcels"] = wide["rainfed_n_parcels"] + wide["irrigated_n_parcels"]
    wide["rainfed_frac"] = wide["rainfed_area_ha"] / wide["total_area_ha"].where(wide["total_area_ha"] > 0)
    return wide.sort_values("rainfed_frac", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    print(f"Reading {args.input} (min parcel = {args.min_parcel_ha} ha) ...")
    out = aggregate(args.input, args.min_parcel_ha, args.chunksize)

    DUN_DIR.mkdir(parents=True, exist_ok=True)
    full_path = DUN_DIR / f"comarca_{args.crop}_rainfed_fraction.csv"
    wl_path = DUN_DIR / f"comarca_{args.crop}_whitelist.csv"

    out.to_csv(full_path, index=False)
    whitelist = out[out["rainfed_frac"] >= args.threshold][["comarca", "rainfed_frac", "total_area_ha"]].copy()
    whitelist.to_csv(wl_path, index=False)

    n_total = len(out)
    n_kept = len(whitelist)
    total_area = out["total_area_ha"].sum()
    kept_area = whitelist["total_area_ha"].sum() if n_kept else 0.0

    print(f"\nWrote {full_path}  ({n_total} comarques)")
    print(f"Wrote {wl_path}     ({n_kept} comarques at >= {args.threshold:.0%} rainfed)")
    print(f"\nArea kept: {kept_area:,.0f} / {total_area:,.0f} ha ({100*kept_area/total_area:.1f}%)")
    print("\nTop / bottom comarques by rainfed fraction:")
    print(out[["comarca", "rainfed_area_ha", "irrigated_area_ha", "rainfed_frac"]].to_string(index=False))


if __name__ == "__main__":
    main()
