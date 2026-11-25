"""Spread each ZIP's demand over several street points and route every site to them.

Run from the project root:
    python3 build_demand_samples.py sample   # writes data/demand_samples.csv
    python3 build_demand_samples.py route    # writes data/travel_times_samples_routed.csv

Sampling: each demand area's polygon (data/modzcta.geojson) is intersected with
ordinary Brooklyn streets from data/nyc-streets.geojson (TIGER S1200 secondary roads and
S1400 local streets; highways, ramps and paths are excluded). The streets are cut into
short pieces, and a length-weighted k-means groups them into SAMPLES_PER_AREA zones.
Each zone is represented by the street point nearest its centre, with weight equal to
the zone's share of street length (a proxy for where calls happen, since there is no
population data below ZIP level). The result covers the whole ZIP, never lands in parks
or water, and is reproducible (fixed seed).

Routing uses the same Valhalla settings and 15 mph conversion as TravelTimes_Routed.ipynb,
so times are comparable with travel_times_routed.csv and travel_times_existing_routed.csv.
Routing is resumable: finished sources are cached in data/.samples_routing_cache.jsonl.
Routing by Valhalla via FOSSGIS. Map data © OpenStreetMap contributors.
"""

import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATA = Path("data")
DEMAND_FILE = DATA / "demand.csv"
POLYGON_FILE = DATA / "modzcta.geojson"
STREET_FILE = DATA / "nyc-streets.geojson"
CANDIDATE_FILE = DATA / "candidate_sites_v2.csv"
EXISTING_FILE = DATA / "FDNY_Firehouse_Listing_20261115.csv"
SAMPLE_FILE = DATA / "demand_samples.csv"
OUTPUT = DATA / "travel_times_samples_routed.csv"
CACHE = DATA / ".samples_routing_cache.jsonl"

BOROUGH = "Brooklyn"
COUNTY_FP = "047"                      # Kings County
STREET_CLASSES = {"S1200", "S1400"}    # secondary roads and local streets
SAMPLES_PER_AREA = 8
PIECE_METERS = 40                      # street length represented by one candidate point
SEED = 15057
KMEANS_ITERATIONS = 60

MATRIX_URL = "https://valhalla1.openstreetmap.de/sources_to_targets"
SPEED_MPH = 15.0
METERS_PER_MILE = 1609.344
MAX_SNAP_METERS = 250
MAX_PAIRS_PER_REQUEST = 100
PAUSE_SECONDS = 1.1


def meters(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_008.8 * math.asin(math.sqrt(min(1.0, a)))


def read_demand():
    with DEMAND_FILE.open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if float(r["calls"]) > 0]
    return {r["area_id"]: float(r["calls"]) for r in rows}


# ---------------------------------------------------------------- sampling
def read_polygons(area_ids):
    polygons = {}
    for feature in json.loads(POLYGON_FILE.read_text())["features"]:
        area = str(feature["properties"]["modzcta"]).strip()
        if area not in area_ids:
            continue
        geometry = feature["geometry"]
        parts = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for part in parts:
            xs = [x for x, _ in part[0]]
            ys = [y for _, y in part[0]]
            polygons.setdefault(area, []).append((min(xs), min(ys), max(xs), max(ys), part))
    missing = area_ids - polygons.keys()
    if missing:
        raise ValueError(f"No polygon for areas: {sorted(missing)}")
    return polygons


def in_ring(x, y, ring):
    inside = False
    for k in range(len(ring)):
        x1, y1 = ring[k - 1]
        x2, y2 = ring[k]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def area_of(lon, lat, polygons):
    for area, parts in polygons.items():
        for x0, y0, x1, y1, part in parts:
            if x0 <= lon <= x1 and y0 <= lat <= y1 and in_ring(lon, lat, part[0]) \
                    and not any(in_ring(lon, lat, hole) for hole in part[1:]):
                return area
    return None


def street_pieces():
    """Yield (lat, lon, length_m) for short pieces of ordinary Brooklyn streets."""
    for feature in json.loads(STREET_FILE.read_text())["features"]:
        props = feature["properties"]
        if props.get("COUNTYFP") != COUNTY_FP or props.get("MTFCC") not in STREET_CLASSES:
            continue
        geometry = feature["geometry"]
        lines = geometry["coordinates"] if geometry["type"] == "MultiLineString" else [geometry["coordinates"]]
        for line in lines:
            for (lon1, lat1), (lon2, lat2) in zip(line, line[1:]):
                length = meters(lat1, lon1, lat2, lon2)
                pieces = max(1, round(length / PIECE_METERS))
                for k in range(pieces):
                    f = (k + 0.5) / pieces
                    yield lat1 + f * (lat2 - lat1), lon1 + f * (lon2 - lon1), length / pieces


def weighted_kmeans(points, k, rng):
    """points: list of (x, y, weight) in local meters. Returns list of k clusters (index lists)."""
    def d2(p, c):
        return (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2

    centers = [points[rng.choices(range(len(points)), weights=[p[2] for p in points])[0]][:2]]
    while len(centers) < k:  # weighted k-means++ initialisation
        scores = [p[2] * min(d2(p, c) for c in centers) for p in points]
        centers.append(points[rng.choices(range(len(points)), weights=scores)[0]][:2])
    for _ in range(KMEANS_ITERATIONS):
        groups = [[] for _ in range(k)]
        for idx, p in enumerate(points):
            groups[min(range(k), key=lambda c: d2(p, centers[c]))].append(idx)
        new_centers = []
        for c, group in enumerate(groups):
            if not group:
                new_centers.append(centers[c])
                continue
            total = sum(points[i][2] for i in group)
            new_centers.append((sum(points[i][0] * points[i][2] for i in group) / total,
                                sum(points[i][1] * points[i][2] for i in group) / total))
        if new_centers == centers:
            break
        centers = new_centers
    return groups, centers


def sample():
    calls = read_demand()
    polygons = read_polygons(set(calls))
    by_area = {a: [] for a in calls}
    for lat, lon, length in street_pieces():
        area = area_of(lon, lat, polygons)
        if area:
            by_area[area].append((lat, lon, length))

    rng = random.Random(SEED)
    rows = []
    for area in sorted(by_area):
        pts = by_area[area]
        if len(pts) < SAMPLES_PER_AREA:
            raise ValueError(f"Area {area} has only {len(pts)} street pieces")
        lat0 = sum(p[0] for p in pts) / len(pts)
        lon0 = sum(p[1] for p in pts) / len(pts)
        kx = 111_320 * math.cos(math.radians(lat0))
        local = [((lon - lon0) * kx, (lat - lat0) * 110_574, w) for lat, lon, w in pts]
        groups, centers = weighted_kmeans(local, SAMPLES_PER_AREA, rng)
        total = sum(w for *_, w in pts)
        for k, (group, center) in enumerate(sorted(zip(groups, centers), key=lambda gc: (-gc[1][1], gc[1][0])), 1):
            if not group:
                continue
            nearest = min(group, key=lambda i: (local[i][0] - center[0]) ** 2 + (local[i][1] - center[1]) ** 2)
            share = sum(pts[i][2] for i in group) / total
            rows.append({"sample_id": f"{area}_{k}", "area_id": area,
                         "lat": round(pts[nearest][0], 6), "lon": round(pts[nearest][1], 6),
                         "weight": share, "calls": calls[area] * share,
                         "street_km": sum(pts[i][2] for i in group) / 1000})
    with SAMPLE_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sample points for {len(by_area)} areas to {SAMPLE_FILE}")


# ---------------------------------------------------------------- routing
def read_sources():
    with CANDIDATE_FILE.open(encoding="utf-8-sig", newline="") as stream:
        sites = [{"id": r["site_id"].strip(), "lat": float(r["lat"]), "lon": float(r["lon"])}
                 for r in csv.DictReader(stream)]
    with EXISTING_FILE.open(encoding="utf-8-sig", newline="") as stream:
        existing = []
        for r in csv.DictReader(stream):  # same filter and numbering as TravelTimes_Routed.ipynb
            if r["Borough"] != BOROUGH or not (r["Latitude"] or "").strip() or not (r["Longitude"] or "").strip():
                continue
            existing.append({"id": f"FH_{len(existing) + 1:03d}",
                             "lat": float(r["Latitude"]), "lon": float(r["Longitude"])})
    return sites + existing


def request_matrix(source, targets):
    location = lambda p: {"lat": p["lat"], "lon": p["lon"], "search_cutoff": MAX_SNAP_METERS}
    payload = {
        "sources": [location(source)],
        "targets": [location(t) for t in targets],
        "costing": "auto",
        "costing_options": {"auto": {"shortest": True, "disable_hierarchy_pruning": True}},
        "units": "kilometers",
        "verbose": True,
    }
    request = Request(MATRIX_URL, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json", "User-Agent": "ems-standby-routing/1.0"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=90) as response:
                result = json.load(response)
            break
        except HTTPError as error:
            if error.code in (429, 502, 503, 504) and attempt < 3:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"Routing failed ({error.code}): {error.read().decode(errors='replace')}") from error
        except (URLError, TimeoutError) as error:
            if attempt < 3:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"Could not reach the routing service: {error}") from error
    matrix = result.get("sources_to_targets")
    if result.get("units") != "kilometers" or not matrix or len(matrix[0]) != len(targets):
        raise ValueError("Router returned an incomplete matrix.")
    return matrix[0]


def route_record(source, target, route):
    pair = f"{source['id']} → {target['sample_id']}"
    km = route.get("distance")
    if km is None or not math.isfinite(km) or km < 0:
        raise ValueError(f"No valid driving distance for {pair}")
    site_snap = meters(source["lat"], source["lon"], route["begin_lat"], route["begin_lon"])
    area_snap = meters(target["lat"], target["lon"], route["end_lat"], route["end_lon"])
    if max(site_snap, area_snap) > MAX_SNAP_METERS + 1:
        raise ValueError(f"Road snap exceeds {MAX_SNAP_METERS} m for {pair}")
    miles = 1000 * km / METERS_PER_MILE
    return {"site_id": source["id"], "sample_id": target["sample_id"], "area_id": target["area_id"],
            "euclid_miles": meters(source["lat"], source["lon"], target["lat"], target["lon"]) / METERS_PER_MILE,
            "network_miles": miles, "minutes": 60 * miles / SPEED_MPH,
            "site_snap_meters": site_snap, "area_snap_meters": area_snap}


def route():
    with SAMPLE_FILE.open(encoding="utf-8", newline="") as stream:
        targets = [{**r, "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(stream)]
    sources = read_sources()
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            entry = json.loads(line)
            done[entry["site_id"]] = entry["records"]
    print(f"{len(sources)} sources × {len(targets)} sample points; {len(done)} sources cached")

    with CACHE.open("a", encoding="utf-8") as cache:
        for n, source in enumerate(sources, 1):
            if source["id"] in done:
                continue
            records = []
            for start in range(0, len(targets), MAX_PAIRS_PER_REQUEST):
                chunk = targets[start:start + MAX_PAIRS_PER_REQUEST]
                row = request_matrix(source, chunk)
                records += [route_record(source, t, r) for t, r in zip(chunk, row)]
                time.sleep(PAUSE_SECONDS)
            done[source["id"]] = records
            cache.write(json.dumps({"site_id": source["id"], "records": records}) + "\n")
            cache.flush()
            if n % 10 == 0 or n == len(sources):
                print(f"Routed {n} of {len(sources)} sources", flush=True)

    rows = [r for s in sources for r in done[s["id"]]]
    if len(rows) != len(sources) * len(targets):
        raise ValueError("Expected exactly one route per source and sample point.")
    tmp = OUTPUT.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(OUTPUT)
    print(f"Wrote {len(rows)} routes to {OUTPUT}")


if __name__ == "__main__":
    {"sample": sample, "route": route}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
