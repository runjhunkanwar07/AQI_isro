from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aqi_isro.demo import DEFAULT_DATA_DIR
from src.aqi_isro.pipeline import load_artifact, load_demo_data, make_aqi_predictions, prepare_hcho_hotspots


def _records_for_json(df, columns):
    safe = df.loc[:, columns].copy()
    for col in safe.columns:
        if str(safe[col].dtype).startswith("datetime64"):
            safe[col] = safe[col].dt.strftime("%Y-%m-%d")
    return safe.to_dict(orient="records")


if __name__ == "__main__":
    output_dir = Path("outputs/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_df, station_df = load_demo_data(DEFAULT_DATA_DIR)
    artifact = load_artifact(DEFAULT_DATA_DIR / "aqi_model.pkl")
    latest_day = spatial_df["date"].max()
    daily_df = spatial_df.loc[spatial_df["date"] == latest_day].copy()
    aqi_df = make_aqi_predictions(daily_df, artifact)
    hotspot_df, summary = prepare_hcho_hotspots(daily_df)

    payload = {
        "date": latest_day.strftime("%Y-%m-%d"),
        "aqi": _records_for_json(aqi_df, ["lat", "lon", "aqi_pred", "aqi_class", "uncertainty"]),
        "hcho": _records_for_json(
            hotspot_df,
            ["lat", "lon", "hcho", "hcho_anomaly", "fire_count", "hotspot_score", "likely_source_region"],
        ),
        "summary": summary,
        "metrics": {**artifact["metrics"], "model_type": artifact.get("model_type", "Model")},
        "feature_importance": artifact["feature_importance"],
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AQI ISRO Hackathon Dashboard</title>
  <style>
    :root {{
      --ink: #17212f;
      --muted: #64748b;
      --paper: #f7fafc;
      --panel: #ffffff;
      --line: #d8e1ec;
      --teal: #0f766e;
      --amber: #d97706;
      --red: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--paper);
      color: var(--ink);
    }}
    header {{
      padding: 26px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .sub {{ color: var(--muted); font-size: 15px; }}
    main {{ padding: 22px 32px 32px; }}
    .toolbar {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 18px;
    }}
    button {{
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      padding: 10px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
    }}
    button.active {{ background: var(--ink); color: #ffffff; border-color: var(--ink); }}
    .grid {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .map {{
      position: relative;
      height: min(66vh, 620px);
      min-height: 420px;
      background:
        linear-gradient(90deg, rgba(15, 118, 110, 0.08) 1px, transparent 1px),
        linear-gradient(rgba(15, 118, 110, 0.08) 1px, transparent 1px),
        #eef7f5;
      background-size: 44px 44px;
      border-radius: 8px;
      overflow: hidden;
    }}
    .point {{
      position: absolute;
      transform: translate(-50%, -50%);
      border-radius: 999px;
      border: 1px solid rgba(23, 33, 47, 0.28);
      box-shadow: 0 1px 5px rgba(23, 33, 47, 0.18);
    }}
    .metric-row {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfdff;
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; }}
    .metric .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 8px 6px;
    }}
    th {{ color: var(--muted); font-weight: 700; }}
    .hidden {{ display: none; }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .grid {{ grid-template-columns: 1fr; }}
      .map {{ min-height: 360px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>AQI and HCHO Intelligence for India</h1>
    <div class="sub">Satellite-data hackathon prototype for {payload["date"]}</div>
  </header>
  <main>
    <div class="toolbar">
      <button id="aqiBtn" class="active" onclick="setMode('aqi')">AQI Map</button>
      <button id="hchoBtn" onclick="setMode('hcho')">HCHO Hotspots</button>
    </div>
    <section class="grid">
      <div class="panel">
        <div id="map" class="map" aria-label="India spatial map"></div>
      </div>
      <aside class="panel">
        <div id="metrics" class="metric-row"></div>
        <table>
          <thead><tr><th>Rank</th><th>Location</th><th>Value</th></tr></thead>
          <tbody id="tableBody"></tbody>
        </table>
      </aside>
    </section>
  </main>
  <script>
    const payload = {json.dumps(payload)};
    const bounds = {{ latMin: 8, latMax: 36.5, lonMin: 68, lonMax: 97.5 }};
    let mode = "aqi";

    function colorAQI(value) {{
      if (value <= 50) return "#1a9850";
      if (value <= 100) return "#91cf60";
      if (value <= 200) return "#fee08b";
      if (value <= 300) return "#fc8d59";
      return "#d73027";
    }}

    function colorHotspot(value) {{
      if (value > 2.6) return "#7f1d1d";
      if (value > 1.9) return "#dc2626";
      if (value > 1.2) return "#f97316";
      if (value > 0.6) return "#facc15";
      return "#0f766e";
    }}

    function xy(row) {{
      const x = ((row.lon - bounds.lonMin) / (bounds.lonMax - bounds.lonMin)) * 100;
      const y = (1 - ((row.lat - bounds.latMin) / (bounds.latMax - bounds.latMin))) * 100;
      return [x, y];
    }}

    function setMode(next) {{
      mode = next;
      document.getElementById("aqiBtn").classList.toggle("active", mode === "aqi");
      document.getElementById("hchoBtn").classList.toggle("active", mode === "hcho");
      render();
    }}

    function renderMetrics() {{
      const metrics = document.getElementById("metrics");
      if (mode === "aqi") {{
        metrics.innerHTML = `
          <div class="metric"><div class="label">RMSE</div><div class="value">${{payload.metrics.rmse.toFixed(2)}}</div></div>
          <div class="metric"><div class="label">R2</div><div class="value">${{payload.metrics.r2.toFixed(2)}}</div></div>
          <div class="metric"><div class="label">Pixels</div><div class="value">${{payload.aqi.length}}</div></div>
          <div class="metric"><div class="label">Model</div><div class="value">${{payload.metrics.model_type || "Model"}}</div></div>
        `;
      }} else {{
        metrics.innerHTML = `
          <div class="metric"><div class="label">Hotspots</div><div class="value">${{payload.summary.hotspot_pixels}}</div></div>
          <div class="metric"><div class="label">Fire link</div><div class="value">${{payload.summary.fire_link_score.toFixed(2)}}</div></div>
          <div class="metric"><div class="label">Mean anomaly</div><div class="value">${{payload.summary.mean_anomaly.toFixed(2)}}</div></div>
          <div class="metric"><div class="label">Top source</div><div class="value" style="font-size:15px">${{payload.summary.top_source_region}}</div></div>
        `;
      }}
    }}

    function renderMap() {{
      const map = document.getElementById("map");
      const rows = mode === "aqi" ? payload.aqi : payload.hcho;
      map.innerHTML = "";
      rows.forEach(row => {{
        const [x, y] = xy(row);
        const point = document.createElement("div");
        point.className = "point";
        point.style.left = `${{x}}%`;
        point.style.top = `${{y}}%`;
        if (mode === "aqi") {{
          point.style.width = `${{8 + row.uncertainty * 0.25}}px`;
          point.style.height = point.style.width;
          point.style.background = colorAQI(row.aqi_pred);
          point.title = `AQI ${{row.aqi_pred.toFixed(1)}} (${{row.aqi_class}})`;
        }} else {{
          point.style.width = `${{7 + row.fire_count}}px`;
          point.style.height = point.style.width;
          point.style.background = colorHotspot(row.hotspot_score);
          point.title = `HCHO score ${{row.hotspot_score.toFixed(2)}} | ${{row.likely_source_region}}`;
        }}
        map.appendChild(point);
      }});
    }}

    function renderTable() {{
      const body = document.getElementById("tableBody");
      const rows = mode === "aqi"
        ? [...payload.aqi].sort((a, b) => b.aqi_pred - a.aqi_pred).slice(0, 10)
        : [...payload.hcho].sort((a, b) => b.hotspot_score - a.hotspot_score).slice(0, 10);
      body.innerHTML = rows.map((row, idx) => {{
        const location = `${{row.lat.toFixed(1)}}, ${{row.lon.toFixed(1)}}`;
        const value = mode === "aqi"
          ? `${{row.aqi_pred.toFixed(1)}} AQI`
          : `${{row.hotspot_score.toFixed(2)}} score`;
        return `<tr><td>${{idx + 1}}</td><td>${{location}}</td><td>${{value}}</td></tr>`;
      }}).join("");
    }}

    function render() {{
      renderMetrics();
      renderMap();
      renderTable();
    }}

    render();
  </script>
</body>
</html>
"""

    out = output_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Static dashboard written to {out}")
