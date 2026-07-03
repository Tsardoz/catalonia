#!/usr/bin/env python3
"""
04_align_phenology.py — Document calendar-DOY phenological anchor windows.

Writes data/processed/phenology_anchors.csv with calendar DOY windows for
five olive phenological stages (S_pre through S8). S5–S8 are sourced from
Garrido et al. 2021 (Forests 12:204, Arbequina BBCH 51–89 in NW Iberia).
S_pre is a pipeline-defined winter/early-vegetative window (DOY 1–71).

Stage overlaps from the Garrido et al. observations are resolved by assigning
shared DOYs to the earlier stage, producing contiguous non-overlapping windows
that cover the full DOY 1–327 range.
"""
import logging
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED,
    S_PRE_DOY, S5_DOY, S6_DOY, S7_DOY, S8_DOY,
    STAGES_RESOLVED,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Stage metadata: labels, BBCH codes, descriptions, and provenance
STAGE_INFO = [
    {
        "stage": "S_pre",
        "label": "pre_inflorescence",
        "bbch": "",
        "description": "Winter rest / early vegetative growth",
        "source": "Pipeline-defined (DOY 1 to S5 start - 1)",
    },
    {
        "stage": "S5",
        "label": "inflorescence",
        "bbch": "51-59",
        "description": "Inflorescence development",
        "source": "Garrido et al. 2021, Forests 12:204 (Arbequina, NW Iberia, 3 seasons)",
    },
    {
        "stage": "S6",
        "label": "flowering",
        "bbch": "61-69",
        "description": "Flowering",
        "source": "Garrido et al. 2021, Forests 12:204",
    },
    {
        "stage": "S7",
        "label": "fruit_dev",
        "bbch": "71-79",
        "description": "Fruit development (includes pit hardening)",
        "source": "Garrido et al. 2021, Forests 12:204",
    },
    {
        "stage": "S8",
        "label": "maturation",
        "bbch": "81-89",
        "description": "Fruit maturation / harvest",
        "source": "Garrido et al. 2021, Forests 12:204",
    },
]


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    raw_stages = {
        "S_pre": S_PRE_DOY,
        "S5": S5_DOY,
        "S6": S6_DOY,
        "S7": S7_DOY,
        "S8": S8_DOY,
    }

    rows = []
    for info in STAGE_INFO:
        stage = info["stage"]
        raw_start, raw_end = raw_stages[stage]
        res_start, res_end = STAGES_RESOLVED[stage]
        rows.append({
            "stage": stage,
            "label": info["label"],
            "bbch": info["bbch"],
            "description": info["description"],
            "doy_start_raw": raw_start,
            "doy_end_raw": raw_end,
            "doy_start": res_start,
            "doy_end": res_end,
            "n_days": res_end - res_start + 1,
            "source": info["source"],
        })

    anchors = pd.DataFrame(rows)
    anchors.to_csv(PROCESSED / "phenology_anchors.csv", index=False)

    log.info("Saved phenology_anchors.csv with %d stages (calendar DOY)", len(anchors))
    for _, row in anchors.iterrows():
        log.info(
            "  %s (%s): DOY %d–%d (%d days)%s",
            row["stage"], row["label"],
            row["doy_start"], row["doy_end"], row["n_days"],
            f"  [BBCH {row['bbch']}]" if row["bbch"] else "",
        )


if __name__ == "__main__":
    main()
