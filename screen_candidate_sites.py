"""Screen proposed ambulance standby points against OpenStreetMap road data.

Run from the project root:
    python screen_candidate_sites.py data/candidate_sites.csv

This is a road-access screen using Valhalla's car profile. Parking permission,
space to stage, and access at the intended hours need separate confirmation.
Map data © OpenStreetMap contributors. https://www.openstreetmap.org/copyright
Routing by Valhalla via FOSSGIS. https://valhalla.github.io/valhalla/
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOCATE_URL = "https://valhalla1.openstreetmap.de/locate"
SEARCH_CUTOFF_M = 250
BATCH_SIZE = 20
AUDIT_FIELDS = [
    "road_screen", "review_reason", "road_snap_meters", "snapped_lat",
    "snapped_lon", "road_name", "road_class", "road_use",
    "parking_aisle_match", "outbound_reach", "osm_way_url", "map_review_url",
]


def read_candidates(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        missing = {"site_id", "lat", "lon"}.difference(fields)
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
        if set(fields).intersection(AUDIT_FIELDS):
            raise ValueError("Use the candidate CSV as input, not the audit CSV.")
        candidates = list(reader)

    if not candidates:
        raise ValueError(f"No candidate rows found in {path}.")
    seen = set()
    for row in candidates:
        site_id = (row["site_id"] or "").strip()
        if not site_id or site_id in seen:
            raise ValueError(f"Blank or duplicate site_id: {site_id!r}")
        seen.add(site_id)
        row["site_id"] = site_id
        try:
            lat, lon = float(row["lat"]), float(row["lon"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{site_id}: lat and lon must be decimal numbers.") from error
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"{site_id}: coordinates are outside the valid range.")
    return fields, candidates


def locate(batch):
    locations = [
        {"lat": float(row["lat"]), "lon": float(row["lon"]), "search_cutoff": SEARCH_CUTOFF_M}
        for row in batch
    ]
    payload = {"locations": locations, "costing": "auto", "verbose": True}
    request = Request(
        LOCATE_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "15C57-Brooklyn-EMS-Study/1.0",
            "X-Client-Id": "15C57-Brooklyn-EMS-Study",
        },
    )
    try:
        with urlopen(request, timeout=45) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"Valhalla returned HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not reach Valhalla: {error}") from error

    if not isinstance(result, list) or len(result) != len(batch):
        raise RuntimeError("Valhalla returned an incomplete list of road matches.")
    for point, match in zip(locations, result):
        for coordinate in ("lat", "lon"):
            if not math.isclose(
                point[coordinate], match[f"input_{coordinate}"], abs_tol=1e-6, rel_tol=0
            ):
                raise RuntimeError("Valhalla returned points in an unexpected order.")
    return result


def screen(row, match, max_snap_m):
    audit = {
        **row,
        "map_review_url": f"https://www.google.com/maps/search/?api=1&query={row['lat']},{row['lon']}",
    }
    edges = [
        edge for edge in (match.get("edges") or [])
        if edge.get("edge", {}).get("access", {}).get("car") is True
        and isinstance(edge.get("distance"), (int, float))
    ]
    if not edges:
        return {**audit, "road_screen": "review", "review_reason": "No mapped car edge found"}

    nearest = min(edges, key=lambda item: item["distance"])
    metadata = nearest["edge"]
    classification = metadata.get("classification") or {}
    road_class = classification.get("classification")
    road_use = classification.get("use")
    info = nearest.get("edge_info") or {}
    way_id = info.get("way_id")
    reasons = []
    if nearest["distance"] > max_snap_m:
        reasons.append(f"Road snap exceeds {max_snap_m:g} m")
    if road_class in {"motorway", "trunk"} or any(
        (classification.get("link"), metadata.get("bridge"), metadata.get("tunnel"),
         metadata.get("round_about"))
    ):
        reasons.append("Near a ramp, major road, bridge, tunnel, or roundabout")
    if metadata.get("destination_only") or metadata.get("access_restriction"):
        reasons.append("Mapped edge has an access restriction")
    if not nearest.get("outbound_reach"):
        reasons.append("Limited or unknown outbound road connectivity")

    return {
        **audit,
        "road_screen": "review" if reasons else "ready_for_routing",
        "review_reason": " | ".join(reasons),
        "road_snap_meters": nearest["distance"],
        "snapped_lat": nearest.get("correlated_lat"),
        "snapped_lon": nearest.get("correlated_lon"),
        "road_name": ", ".join(info.get("names") or []),
        "road_class": road_class,
        "road_use": road_use,
        "parking_aisle_match": road_use == "parking_aisle",
        "outbound_reach": nearest.get("outbound_reach"),
        "osm_way_url": f"https://www.openstreetmap.org/way/{way_id}" if way_id else "",
    }


def write_csv(path, fields, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path("data/candidate_sites.csv"))
    parser.add_argument("--max-snap-m", type=float, default=30, help="Review cutoff in meters (default 30).")
    args = parser.parse_args()
    if not 0 < args.max_snap_m <= SEARCH_CUTOFF_M:
        parser.error(f"--max-snap-m must be greater than 0 and at most {SEARCH_CUTOFF_M}.")

    try:
        fields, candidates = read_candidates(args.input)
        audit = []
        for offset in range(0, len(candidates), BATCH_SIZE):
            if offset:
                time.sleep(1.1)
            batch = candidates[offset : offset + BATCH_SIZE]
            audit.extend(screen(row, match, args.max_snap_m) for row, match in zip(batch, locate(batch)))

        ready = [row for row in audit if row["road_screen"] == "ready_for_routing"]
        audit_path = args.input.with_name(f"{args.input.stem}_audit.csv")
        screened_path = args.input.with_name(f"{args.input.stem}_screened.csv")
        write_csv(audit_path, fields + AUDIT_FIELDS, audit)
        write_csv(screened_path, fields, ({field: row[field] for field in fields} for row in ready))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Candidate screening stopped: {error}\n")

    print(f"{len(ready)} ready for routing; {len(audit) - len(ready)} need review.")
    print(f"All candidates with audit details: {audit_path}")
    print(f"Road-screened candidate list: {screened_path}")
    if len(ready) < 5:
        print("Review or replace flagged points to get at least five candidates before solving.")


if __name__ == "__main__":
    main()
