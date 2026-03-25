"""
Download Sentinel-2 NIR (B08) and SWIR1 (B11) bands for deforestation detection.

Uses pystac-client to search the Element84 Earth Search STAC catalog and
rasterio windowed reads (HTTP range requests on COGs) to download only the
AOI extent. B08 (10m) is resampled to 20m to match B11.

Only downloads scenes where >= 85% of the AOI has clear pixels (no clouds,
shadows, nodata). SCL is downloaded first to check; B08/B11 are skipped
for unusable dates.

Output: Data/bands/YYYY-MM-DD_B08.tif  (NIR, resampled to 20m)
        Data/bands/YYYY-MM-DD_B11.tif  (SWIR1, 20m native)
        Data/bands/YYYY-MM-DD_SCL.tif  (Scene Classification, 20m)
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio # type: ignore
from rasterio.enums import Resampling  # type: ignore
from rasterio.transform import from_bounds  # type: ignore
from rasterio.warp import transform_bounds  # type: ignore
from rasterio.windows import from_bounds as window_from_bounds  # type: ignore
from pystac_client import Client  # type: ignore

# ── Config ──────────────────────────────────────────────────────────────────
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# AOI bounding box [west, south, east, north]
BBOX = [-52.5816, -4.0448, -52.2242, -3.7550]

DATE_RANGE = "2018-01-01/2026-03-14"

OUT_DIR = Path("Data/bands")

# Target resolution in metres (B11 native)
TARGET_RES = 20

# Minimum fraction of clear pixels to keep a scene
MIN_CLEAR_FRACTION = 0.80

# SCL classes that are unusable
BAD_SCL = {0, 1, 2, 3, 8, 9, 10, 11}


def search_scenes():
    """Search STAC catalog and return items grouped by date (best cloud cover)."""
    print("Searching STAC catalog …")
    client = Client.open(STAC_URL)
    search = client.search(
        collections=[COLLECTION],
        bbox=BBOX,
        datetime=DATE_RANGE,
        max_items=None,
    )
    items = list(search.items())
    print(f"  Found {len(items)} items")

    # Group by date, keep lowest cloud cover per date
    by_date: dict[str, list] = defaultdict(list)
    for item in items:
        date_str = item.datetime.strftime("%Y-%m-%d")
        by_date[date_str].append(item)

    best = {}
    for date_str, group in sorted(by_date.items()):
        winner = min(group, key=lambda it: it.properties.get("eo:cloud_cover", 100))
        best[date_str] = winner

    print(f"  {len(best)} unique dates after deduplication")
    return best


def read_band(href, resampling=Resampling.bilinear):
    """Read a single band clipped to the AOI, resampled to 20m. Returns (data, profile)."""
    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *BBOX)
        win = window_from_bounds(left, bottom, right, top, src.transform)

        out_width = max(1, int(round((right - left) / TARGET_RES)))
        out_height = max(1, int(round((top - bottom) / TARGET_RES)))

        data = src.read(1, window=win, out_shape=(out_height, out_width), resampling=resampling)

        out_transform = from_bounds(left, bottom, right, top, out_width, out_height)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=out_height,
            width=out_width,
            count=1,
            transform=out_transform,
            crs=src.crs,
            compress="deflate",
        )
    return data, profile


def save_tif(path, data, profile):
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def scene_is_clear(scl_data):
    """Check if >= MIN_CLEAR_FRACTION of pixels are usable."""
    total = scl_data.size
    clear = np.count_nonzero(~np.isin(scl_data, list(BAD_SCL)))
    return clear / total >= MIN_CLEAR_FRACTION, clear / total


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scenes = search_scenes()
    total = len(scenes)
    downloaded = 0
    skipped = 0

    for i, (date_str, item) in enumerate(sorted(scenes.items()), 1):
        prefix = f"[{i}/{total}] {date_str}"

        # Check if all three files already exist (previously downloaded & passed)
        b08_path = OUT_DIR / f"{date_str}_B08.tif"
        b11_path = OUT_DIR / f"{date_str}_B11.tif"
        scl_path = OUT_DIR / f"{date_str}_SCL.tif"

        if b08_path.exists() and b11_path.exists() and scl_path.exists():
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"  {prefix} … exists (kept {downloaded} so far)")
            continue

        # Step 1: Download SCL first to check quality
        if "scl" not in item.assets:
            print(f"  {prefix} … no SCL asset, skipping")
            skipped += 1
            continue

        try:
            scl_data, scl_profile = read_band(item.assets["scl"].href, Resampling.nearest)
        except Exception as e:
            print(f"  {prefix} … SCL read error: {e}")
            skipped += 1
            continue

        is_clear, clear_frac = scene_is_clear(scl_data)
        if not is_clear:
            skipped += 1
            if skipped % 20 == 0:
                print(f"  {prefix} … {clear_frac:.0%} clear, skip (skipped {skipped} so far)")
            continue

        # Step 2: Scene is good — download B08 and B11
        try:
            b08_data, b08_profile = read_band(item.assets["nir"].href, Resampling.bilinear)
            b11_data, b11_profile = read_band(item.assets["swir16"].href, Resampling.bilinear)
        except Exception as e:
            print(f"  {prefix} … band read error: {e}")
            skipped += 1
            continue

        # Extra check: B08 should have actual data (not all zeros / off-tile)
        if np.count_nonzero(b08_data) / b08_data.size < MIN_CLEAR_FRACTION:
            skipped += 1
            continue

        save_tif(scl_path, scl_data, scl_profile)
        save_tif(b08_path, b08_data, b08_profile)
        save_tif(b11_path, b11_data, b11_profile)
        downloaded += 1
        print(f"  {prefix} … ok ({clear_frac:.0%} clear, kept {downloaded})")

    print(f"\nDone. Kept {downloaded}, skipped {skipped}.")


if __name__ == "__main__":
    main()
