"""
dun_cultivar_fraction.py
Per-comarca cultivar-share table for a single crop and irrigation regime
from a DUN parcel CSV (e.g. olives.csv from analisi.transparenciacatalunya.cat).

For each comarca, restricted to parcels with SIST_EXPL == --regime ('S' rainfed
or 'R' irrigated) in --campaign:
  total_ha   = sum SUP_NETA_H
  cult_ha    = sum SUP_NETA_H over parcels whose NVAR contains --cultivar
  cult_frac  = cult_ha / total_ha

Parcels below MIN_PARCEL_HA are dropped before aggregation. Comarca name
normalisation matches src/parse_yield.py.

CLI:
  python src/dun_cultivar_fraction.py --crop olive --regime R --cultivar ARBEQUINA \
      --campaign 2024 --threshold 0.90 --min-cult-ha 500 \
      --input data/dun/olives.csv

Outputs (suffix follows --regime: rainfed for S, irrigated for R):
  data/dun/comarca_{regime}_{cultivar}_fraction.csv
  data/dun/comarca_{regime}_{cultivar}_whitelist.csv
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DUN_DIR = DATA_DIR / "dun"

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
    "Nom de la varietat del cultiu (NVAR)",
]

RENAME = {
    "Campanya (CAMPANYA)":                              "year",
    "Nom comarca (NOM_COM)":                            "comarca",
    "Superf\u00edcie neta en hect\u00e0rees (SUP_NETA_H)": "sup_neta_h",
    "Sistema d\u2019explotaci\u00f3 (SIST_EXPL)":       "sist",
    "Nom de la varietat del cultiu (NVAR)":             "nvar",
}

REGIME_LABEL = {"S": "rainfed", "R": "irrigated"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop", required=True, help="Short crop name used in output filenames, e.g. 'olive'")
    p.add_argument("--input", type=Path, required=True, help="Path to DUN parcel CSV for this crop")
    p.add_argument("--regime", choices=["S", "R"], required=True, help="SIST_EXPL value: S=rainfed, R=irrigated")
    p.add_argument("--cultivar", required=True, help="Substring matched (case-insensitive) against NVAR, e.g. ARBEQUINA")
    p.add_argument("--campaign", default="2024", help="Single CAMPANYA year to use as the structural snapshot (default 2024)")
    p.add_argument("--min-parcel-ha", type=float, default=0.5, help="Minimum SUP_NETA_H per parcel (default 0.5)")
    p.add_argument("--threshold", type=float, default=0.90, help="Cultivar-fraction whitelist cutoff (default 0.90)")
    p.add_argument("--min-cult-ha", type=float, default=500.0, help="Minimum cultivar ha for whitelist (default 500)")
    p.add_argument("--chunksize", type=int, default=200_000, help="CSV read chunk size")
    return p.parse_args()


def aggregate(input_path: Path, regime: str, cultivar: str, campaign: str,
              min_parcel_ha: float, chunksize: int) -> pd.DataFrame:
    cult_upper = cultivar.upper()
    parts = []
    reader = pd.read_csv(input_path, usecols=USECOLS, dtype=str, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        chunk = chunk.rename(columns=RENAME)
        chunk = chunk[chunk["year"] == campaign]
        if chunk.empty:
            continue
        chunk["sup_neta_h"] = pd.to_numeric(
            chunk["sup_neta_h"].str.replace(",", ".", regex=False), errors="coerce"
        )
        chunk = chunk.dropna(subset=["sup_neta_h", "comarca", "sist"])
        chunk = chunk[chunk["sup_neta_h"] >= min_parcel_ha]
        chunk = chunk[chunk["sist"] == regime]
        chunk["comarca"] = chunk["comarca"].str.strip().replace(COMARCA_NORMALISE)
        chunk["nvar"] = chunk["nvar"].fillna("").str.upper()
        chunk["is_cult"] = chunk["nvar"].str.contains(cult_upper, regex=False)
        parts.append(chunk[["comarca", "sup_neta_h", "is_cult"]])

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["comarca", "sup_neta_h", "is_cult"])

    out = df.groupby("comarca").agg(
        total_ha=("sup_neta_h", "sum"),
        n_parcels=("sup_neta_h", "size"),
        cult_ha=("sup_neta_h", lambda s: s[df.loc[s.index, "is_cult"]].sum()),
        cult_n_parcels=("is_cult", "sum"),
    ).reset_index()
    out["cult_frac"] = out["cult_ha"] / out["total_ha"].where(out["total_ha"] > 0)
    return out.sort_values("cult_ha", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    regime_lbl = REGIME_LABEL[args.regime]
    cult_lbl = args.cultivar.lower()
    print(f"Reading {args.input} (campaign={args.campaign}, regime={args.regime}={regime_lbl}, "
          f"cultivar~={args.cultivar}, min parcel = {args.min_parcel_ha} ha) ...")
    out = aggregate(args.input, args.regime, args.cultivar, args.campaign,
                    args.min_parcel_ha, args.chunksize)

    DUN_DIR.mkdir(parents=True, exist_ok=True)
    full_path = DUN_DIR / f"comarca_{regime_lbl}_{cult_lbl}_fraction.csv"
    wl_path = DUN_DIR / f"comarca_{regime_lbl}_{cult_lbl}_whitelist.csv"

    out.to_csv(full_path, index=False)
    keep = (out["cult_frac"] >= args.threshold) & (out["cult_ha"] >= args.min_cult_ha)
    whitelist = out.loc[keep, ["comarca", "cult_frac", "cult_ha", "total_ha"]].copy()
    whitelist = whitelist.rename(columns={
        "cult_frac": f"{cult_lbl}_frac",
        "cult_ha":   f"{cult_lbl}_ha",
    })
    whitelist.to_csv(wl_path, index=False)

    n_total = len(out)
    n_kept = len(whitelist)
    total_area = out["total_ha"].sum()
    kept_area = whitelist[f"{cult_lbl}_ha"].sum() if n_kept else 0.0

    print(f"\nWrote {full_path}  ({n_total} comarques with any {regime_lbl} {args.crop})")
    print(f"Wrote {wl_path}     ({n_kept} comarques at >= {args.threshold:.0%} {args.cultivar} "
          f"AND >= {args.min_cult_ha:g} ha)")
    print(f"\n{args.cultivar} area kept: {kept_area:,.0f} / {total_area:,.0f} ha "
          f"({100*kept_area/total_area:.1f}% of all {regime_lbl} {args.crop})")
    print(f"\nFull table:")
    print(out.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
