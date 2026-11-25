"""Mean and standard deviation of weekly EMS call counts per ZIP area and calendar quarter (2025).

Weeks run Monday-Friday (the data is weekdays only). A week cut by a quarter boundary is kept
within its quarter, scaled to 5 weekdays, and weighted by the number of weekdays it covers, so a
1-day fragment counts one-fifth as much as a full week. ZIP codes are grouped into the 37 model
areas the same way as data/call_weights_by_zip.csv.
"""
import csv, sys, math, collections
from datetime import date, timedelta

calls_file, weights_file, out_file = sys.argv[1:4]
MERGE = {"11249": "11211", "11251": "11211", "11241": "11201", "11242": "11201"}   # as in call_weights_by_zip.csv
areas = {r["zipcode"] for r in csv.DictReader(open(weights_file))}

quarter = lambda d: (d.month - 1) // 3 + 1
week_key = lambda d: (quarter(d), d - timedelta(days=d.weekday()))          # (quarter, Monday of the week)

# every weekday of 2025, grouped into quarter-weeks
days_in = collections.Counter()
d = date(2025, 1, 1)
while d.year == 2025:
    if d.weekday() < 5: days_in[week_key(d)] += 1
    d += timedelta(days=1)

counts = collections.Counter()          # (area, quarter, week) -> calls
dates_seen = set()
for row in csv.DictReader(open(calls_file)):
    z = MERGE.get(row["zipcode"], row["zipcode"])
    if z not in areas: continue
    dt = date.fromisoformat(row["incident_datetime"][:10]); dates_seen.add(dt)
    counts[(z, *week_key(dt))] += 1

rows = []
for z in sorted(areas):
    for q in (1, 2, 3, 4):
        weeks = sorted(k for k in days_in if k[0] == q)
        n_days = [days_in[k] for k in weeks]
        rate = [counts[(z, *k)] * 5 / nd for k, nd in zip(weeks, n_days)]        # calls per 5-weekday week
        wsum = sum(n_days)
        mean = sum(r * nd for r, nd in zip(rate, n_days)) / wsum
        # weighted sample variance (reliability weights), weights = weekdays covered
        v1, v2 = wsum, sum(nd * nd for nd in n_days)
        var = sum(nd * (r - mean) ** 2 for r, nd in zip(rate, n_days)) / (v1 - v2 / v1)
        total = sum(counts[(z, *k)] for k in weeks)
        rows.append({"area_id": z, "quarter": f"Q{q}", "weekdays": wsum, "weeks": round(wsum / 5, 1),
                     "total_calls": total, "mean_weekly_calls": round(mean, 2), "std_weekly_calls": round(math.sqrt(var), 2),
                     "cv": round(math.sqrt(var) / mean, 3) if mean else None})
with open(out_file, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("weekdays in 2025:", sum(days_in.values()), "| weekdays with calls:", len(dates_seen))
