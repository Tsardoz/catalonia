#!/usr/bin/env python3
"""
04_align_phenology.py — Create phenological anchor windows.

Anchors olive phenological stages to literature GDD values (not fit from data).
Writes data/processed/phenology_anchors.csv with source column for traceability.
"""
import logging
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, PRE_FLOWERING_GDD, FLOWERING_GDD, PIT_HARDENING_GDD, OIL_ACCUMULATION_GDD,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    anchors = pd.DataFrame([
        {
            "stage": "pre_flowering",
            "gdd_start": PRE_FLOWERING_GDD[0],
            "gdd_end": PRE_FLOWERING_GDD[1],
            "cultivar_group": "Arbequina (general)",
            "source": "Sanz-Cortés et al. 2002 (Ann Appl Biol 140:151, BBCH 50-59); "
                      "Didevarasl et al. 2023 (Plants 12:3181, sprouting phase)",
            "notes": "Bud break through inflorescence development, T_b=10°C",
        },
        {
            "stage": "flowering",
            "gdd_start": FLOWERING_GDD[0],
            "gdd_end": FLOWERING_GDD[1],
            "cultivar_group": "Arbequina (general)",
            "source": "Didevarasl et al. 2023 (Plants 12:3181); "
                      "Sanz-Cortés et al. 2002 (Ann Appl Biol 140:151)",
            "notes": "Mid-late cultivar blooming GDD, T_b=10°C",
        },
        {
            "stage": "pit_hardening",
            "gdd_start": PIT_HARDENING_GDD[0],
            "gdd_end": PIT_HARDENING_GDD[1],
            "cultivar_group": "Arbequina (general)",
            "source": "Didevarasl et al. 2023 (Plants 12:3181); "
                      "Rapoport et al. 2013 (Ann Appl Biol 163:200)",
            "notes": "Pit sclerification period, T_b=10°C",
        },
        {
            "stage": "oil_accumulation",
            "gdd_start": OIL_ACCUMULATION_GDD[0],
            "gdd_end": OIL_ACCUMULATION_GDD[1],
            "cultivar_group": "Arbequina (general)",
            "source": "López-Bernal et al. 2021 (Eur J Agron 123:126206); "
                      "Siakou et al. 2022 (Agronomy 12:879)",
            "notes": "Oil accumulation follows pit hardening, T_b=10°C",
        },
    ])

    anchors.to_csv(PROCESSED / "phenology_anchors.csv", index=False)
    log.info(f"Saved phenology_anchors.csv with {len(anchors)} stages")
    for _, row in anchors.iterrows():
        log.info(f"  {row['stage']}: {row['gdd_start']}-{row['gdd_end']} GDD")


if __name__ == "__main__":
    main()
