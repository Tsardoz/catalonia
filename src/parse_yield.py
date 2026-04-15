"""
parse_yield.py  —  Step 1
Parse rainfed (secà) woody crop yield data from Produccions_comarcals xlsx files.

Uses openpyxl directly to handle merged cells correctly. pandas read_excel
misassigns comarca names on these files due to merged cell handling.

Output CSV columns:
    year, comarca, crop_catalan, crop_en, crop_group, pheno_key,
    seca_ha, seca_kg_ha, yield_tha
"""

import re
from pathlib import Path

import openpyxl
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
SHEET = "LLENYOSOS"

# Columns by position in the raw LLENYOSOS sheet
COL_COMARCA   = 0
COL_GRUP      = 1
COL_CROP      = 2
COL_SECA_HA   = 3
COL_SECA_KGHA = 5  # rainfed yield kg/ha (col 4 is irrigated area)

# target crops: catalan name -> (english name, crop_group)
# double-space variant of raisin grape is included as a separate key
TARGET_CROPS = {
    "Pomera":                       ("Apple",                  "pome_fruit"),
    "Perera":                       ("Pear",                   "pome_fruit"),
    "Presseguer":                   ("Peach",                  "stone_fruit"),
    "Pruner":                       ("Plum",                   "stone_fruit"),
    "Albercoquer":                  ("Apricot",                "stone_fruit"),
    "Nectariner":                   ("Nectarine",              "stone_fruit"),
    "Prèssec plà":                  ("Flat peach",             "stone_fruit"),
    "Platerina":                    ("Flat nectarine",         "stone_fruit"),
    "Cirerer i guinder":            ("Cherry and sour cherry", "stone_fruit"),
    "Ametller":                     ("Almond",                 "nut"),
    "Avellaner":                    ("Hazelnut",               "nut"),
    "Noguera":                      ("Walnut",                 "nut"),
    "Vinya de raïm per a vi":       ("Wine grape",             "vine"),
    "Vinya de raïm per a panses":   ("Raisin grape",           "vine"),
    "Vinya de raïm per a  panses":  ("Raisin grape",           "vine"),   # double-space variant
    "Olivera per a oliva d'oli":    ("Oil olive",              "olive"),
    "Magraner":                     ("Pomegranate",            "other"),
    "Caqui":                        ("Persimmon",              "other"),
    "Nashi":                        ("Asian pear",             "other"),
}

# Comarca name normalisation — fixes naming inconsistencies across source years
COMARCA_NORMALISE = {
    "La Selva":       "Selva",
    "la Selva":       "Selva",
    "Vall d'Aran":    "Aran",
    "Pla d 'Urgell":  "Pla d'Urgell",   # space before apostrophe typo
}

# pheno_key overrides for crops that get crop-specific phenological windows
PHENO_KEY_OVERRIDES = {
    "Cirerer i guinder": "cherry",
    "Ametller":          "almond",
    "Albercoquer":       "apricot",
    "Noguera":           "walnut",
    "Avellaner":         "hazelnut",
}


def _pheno_key(crop_catalan: str, crop_group: str) -> str:
    return PHENO_KEY_OVERRIDES.get(crop_catalan, crop_group)


def parse_one_year(path: Path) -> pd.DataFrame:
    year = int(re.search(r"(\d{4})", path.stem).group(1))
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[SHEET]

    records = []
    comarca = grup = None

    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[COL_COMARCA] is not None:
            comarca = str(row[COL_COMARCA]).strip()
            comarca = COMARCA_NORMALISE.get(comarca, comarca)
        if row[COL_GRUP] is not None:
            grup = str(row[COL_GRUP]).strip()

        raw_crop = row[COL_CROP]
        if raw_crop is None or comarca is None:
            continue

        crop = str(raw_crop).strip()
        if crop not in TARGET_CROPS:
            continue

        try:
            seca_ha   = float(row[COL_SECA_HA])   if row[COL_SECA_HA]   is not None else 0.0
            seca_kgha = float(row[COL_SECA_KGHA]) if row[COL_SECA_KGHA] is not None else 0.0
        except (TypeError, ValueError):
            continue

        if seca_ha <= 0 or seca_kgha <= 0:
            continue

        crop_en, crop_group = TARGET_CROPS[crop]

        records.append({
            "year":         year,
            "comarca":      comarca,
            "crop_catalan": crop,
            "crop_en":      crop_en,
            "crop_group":   crop_group,
            "pheno_key":    _pheno_key(crop, crop_group),
            "seca_ha":      seca_ha,
            "seca_kg_ha":   seca_kgha,
            "yield_tha":    seca_kgha / 1000.0,
        })

    wb.close()
    return pd.DataFrame(records)


def load_all_years() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("Produccions_comarcals_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No Produccions_comarcals files found in {DATA_DIR}")
    frames = [parse_one_year(f) for f in files]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_all_years()
    out = DATA_DIR / "catalan_woody_yield_raw.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} records to {out}")
    print(f"Years:     {sorted(df['year'].unique())}")
    print(f"Comarques: {df['comarca'].nunique()}")
    print(f"\nRecords per crop_group:")
    print(df.groupby("crop_group")["yield_tha"].count().to_string())
    print(f"\nSample:")
    print(df.head(10).to_string(index=False))
