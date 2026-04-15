"""
download_agera5.py  —  Step 3
Download AgERA5 agrometeorological indicators for Catalonia (2015–2024) via CDS API.

Three request types:
  - SINGLE_VARIABLES: no statistic, version="2_0"
                      (reference_evapotranspiration, precipitation_flux)
  - MEAN_VARIABLES:   statistic="24_hour_mean", version="1_0" with v2_0 fallback
                      (vapour_pressure — v1.0 cuts off mid-2023; 2023–2024 use v2.0)
  - Temperature:      variable="2m_temperature" + statistic + version="2_0"

Version requirements were determined empirically — the CDS MultiAdaptor returns
MultiAdaptorNoDataError if the wrong version is used for a given variable.

The new CDS API delivers one NetCDF per day inside a ZIP archive.
This script downloads the ZIP, extracts the daily files, concatenates
them into a single annual NetCDF, then deletes the ZIP and temp files.

Resume-safe: skips files that are already valid NetCDF4.
Requires ~/.cdsapirc with valid CDS credentials.
"""

import os
import tempfile
import time
import zipfile
from pathlib import Path

os.environ["TQDM_DISABLE"] = "1"  # suppress all tqdm bars (xarray, cdsapi)

import cdsapi
import xarray as xr


def _fmt(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


class Progress:
    """Tracks download progress and prints an ETA after each file."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0          # files completed (downloaded or skipped)
        self.downloaded = 0    # files actually fetched (excludes skips)
        self._start = time.time()
        self._fetch_time = 0.0 # cumulative seconds spent on real downloads

    def tick(self, skipped: bool = False, elapsed_file: float = 0.0) -> None:
        self.done += 1
        if not skipped:
            self.downloaded += 1
            self._fetch_time += elapsed_file

        remaining = self.total - self.done
        total_elapsed = time.time() - self._start

        if self.downloaded > 0:
            avg = self._fetch_time / self.downloaded
            eta_secs = avg * remaining
            eta_str = _fmt(eta_secs)
            finish = time.localtime(time.time() + eta_secs)
            finish_str = time.strftime("%H:%M", finish)
        else:
            eta_str = "?"
            finish_str = "?"

        if remaining > 0:
            suffix = f", ~{eta_str} remaining (done ~{finish_str})"
        else:
            suffix = " — all done!"

        print(
            f"  [{self.done}/{self.total}] elapsed {_fmt(total_elapsed)}{suffix}",
            flush=True,
        )

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "agera5_catalonia"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "sis-agrometeorological-indicators"
YEARS   = [str(y) for y in range(2015, 2025)]
MONTHS  = [f"{m:02d}" for m in range(1, 13)]
DAYS    = [f"{d:02d}" for d in range(1, 32)]
# Catalonia bounding box [North, West, South, East]
AREA = [42.51, -0.09, 40.31, 3.19]

# Variables with no statistic parameter. version="2_0" required.
SINGLE_VARIABLES = [
    "reference_evapotranspiration",
    "precipitation_flux",
]

# Variables that require statistic="24_hour_mean".
# vapour_pressure (hPa) is the actual vapour pressure; divide by 10 → kPa for VPD.
MEAN_VARIABLES = [
    "vapour_pressure",
]

# 2m_temperature statistics (version="2_0" required).
# Valid values per CDS constraints: 24_hour_maximum, 24_hour_minimum,
#   day_time_maximum, day_time_mean, night_time_mean, night_time_minimum.
# Key = statistic sent to API; Value = suffix used in output filename.
TEMP_STATISTICS = {
    "24_hour_maximum": "max",
    "24_hour_minimum": "min",
}

SLEEP_BETWEEN = 2  # seconds — be polite to the API


def _is_valid_netcdf(path: Path) -> bool:
    """Return True if path is a valid NetCDF4/HDF5 or NetCDF3 file."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic in (b"\x89HDF", b"CDF\x01", b"CDF\x02")
    except OSError:
        return False


def _zip_to_netcdf(zip_path: Path, output_nc: Path) -> None:
    """
    Extract daily NetCDF files from a CDS ZIP and concatenate into
    a single annual NetCDF with a proper time dimension.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"    Extracting ZIP ({zip_path.name}) ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmpdir)
        nc_files = sorted(Path(tmpdir).glob("*.nc"))
        if not nc_files:
            raise RuntimeError(f"No .nc files found inside {zip_path.name}")
        print(f"    Concatenating {len(nc_files)} daily files ...")
        ds = xr.open_mfdataset(nc_files, engine="netcdf4", combine="by_coords")
        ds.to_netcdf(output_nc)
        ds.close()
    print(f"    Saved {output_nc.name}")


def _base_request(version: str | None = None) -> dict:
    req = {
        "year":  YEARS,
        "month": MONTHS,
        "day":   DAYS,
        "area":  AREA,
    }
    if version is not None:
        req["version"] = version
    return req


def _fetch_and_consolidate(
    client: cdsapi.Client,
    req: dict,
    output_nc: Path,
    label: str,
    progress: Progress,
) -> None:
    """
    Download a CDS request to a temp ZIP, consolidate daily NetCDFs into
    a single annual NetCDF, then clean up.

    If output_nc already exists and is a valid NetCDF: skip.
    If output_nc exists but is a ZIP (from a previous failed run): consolidate.
    If output_nc doesn't exist: download then consolidate.
    """
    # Already a valid NetCDF — nothing to do
    if output_nc.exists() and _is_valid_netcdf(output_nc):
        print(f"  Skip {output_nc.name} — already valid NetCDF")
        progress.tick(skipped=True)
        return

    t0 = time.time()

    # Existing file is a ZIP (e.g. from a previous partial run) — consolidate directly
    if output_nc.exists() and zipfile.is_zipfile(output_nc):
        print(f"  Processing existing ZIP: {output_nc.name}")
        try:
            _zip_to_netcdf(output_nc, output_nc.with_suffix(".nc.tmp"))
            output_nc.with_suffix(".nc.tmp").replace(output_nc)
        except Exception as e:
            print(f"  FAILED consolidating {output_nc.name}: {e}")
        progress.tick(skipped=False, elapsed_file=time.time() - t0)
        return

    # Download fresh
    zip_path = output_nc.with_suffix(".zip")
    print(f"  Requesting {label} ...")
    try:
        client.retrieve(DATASET, req).download(str(zip_path))
        _zip_to_netcdf(zip_path, output_nc)
        zip_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"  FAILED {label}: {e}")
        zip_path.unlink(missing_ok=True)

    progress.tick(skipped=False, elapsed_file=time.time() - t0)
    time.sleep(SLEEP_BETWEEN)


def download_single(client: cdsapi.Client, variable: str, year: str, progress: Progress) -> None:
    """Download a variable that needs no statistic parameter (version="2_0" required)."""
    output_nc = OUTPUT_DIR / f"{variable}_{year}.nc"
    req = {**_base_request(version="2_0"), "variable": variable, "year": [year]}
    _fetch_and_consolidate(client, req, output_nc, f"{variable} {year}", progress)


def download_mean(client: cdsapi.Client, variable: str, year: str, progress: Progress) -> None:
    """
    Download a variable with statistic='24_hour_mean'.
    Tries version='1_0' first; falls back to '2_0' if the file produced is
    incomplete (<300 daily files, e.g. vapour_pressure v1_0 cuts off mid-2023)
    or if the request fails outright.
    """
    output_nc = OUTPUT_DIR / f"{variable}_{year}.nc"

    # If an existing file is valid but might be incomplete (v1_0 cutoff)
    # re-check by looking at time dimension length
    if output_nc.exists() and _is_valid_netcdf(output_nc):
        try:
            ds = xr.open_dataset(output_nc, engine="netcdf4")
            n_days = ds.sizes.get("time", 0)
            ds.close()
            if n_days >= 300:  # full year
                print(f"  Skip {output_nc.name} — already valid NetCDF ({n_days} days)")
                progress.tick(skipped=True)
                return
            else:
                print(f"  {output_nc.name} incomplete ({n_days} days) — re-fetching with v2_0")
                output_nc.unlink()
        except Exception:
            pass

    for version in ("1_0", "2_0"):
        req = {**_base_request(version=version), "variable": variable,
               "statistic": "24_hour_mean", "year": [year]}
        zip_path = output_nc.with_suffix(".zip")
        print(f"  Requesting {variable} {year} (v{version}) ...")
        t0 = time.time()
        try:
            client.retrieve(DATASET, req).download(str(zip_path))
            _zip_to_netcdf(zip_path, output_nc)
            zip_path.unlink(missing_ok=True)
            progress.tick(skipped=False, elapsed_file=time.time() - t0)
            time.sleep(SLEEP_BETWEEN)
            return
        except Exception as e:
            zip_path.unlink(missing_ok=True)
            print(f"  FAILED {variable} {year} v{version}: {str(e).split(chr(10))[0]}")

    progress.tick(skipped=False, elapsed_file=0)
    print(f"  Giving up on {variable} {year}")


def download_temperature(client: cdsapi.Client, statistic: str, suffix: str, year: str, progress: Progress) -> None:
    """Download 2m_temperature with a specific statistic (version 2_0 required)."""
    output_nc = OUTPUT_DIR / f"2m_temperature_{suffix}_{year}.nc"
    req = {
        **_base_request(version="2_0"),
        "variable":  "2m_temperature",
        "statistic": statistic,
        "year":      [year],
    }
    _fetch_and_consolidate(client, req, output_nc, f"2m_temperature ({statistic}) {year}", progress)


if __name__ == "__main__":
    client = cdsapi.Client(quiet=True)

    total = (
        len(SINGLE_VARIABLES) * len(YEARS)
        + len(MEAN_VARIABLES) * len(YEARS)
        + len(TEMP_STATISTICS) * len(YEARS)
    )
    progress = Progress(total)

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Years: {YEARS[0]}–{YEARS[-1]}")
    print(f"Total files: {total}")
    print()

    print("=== Single variables (no statistic) ===")
    for variable in SINGLE_VARIABLES:
        print(f"\n{variable}")
        for year in YEARS:
            download_single(client, variable, year, progress)

    print("\n=== Mean variables (statistic=24_hour_mean) ===")
    for variable in MEAN_VARIABLES:
        print(f"\n{variable}")
        for year in YEARS:
            download_mean(client, variable, year, progress)

    print("\n=== Temperature (version 2_0) ===")
    for statistic, suffix in TEMP_STATISTICS.items():
        print(f"\n2m_temperature / {statistic}")
        for year in YEARS:
            download_temperature(client, statistic, suffix, year, progress)

    print("\nDone.")
