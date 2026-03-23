let map;
let geoLayer;
let forecastByHorizon = {};
let tablesByHorizon = {};
let countyGeojson = null;

const horizonSelect = document.getElementById("horizonSelect");
const metricSelect = document.getElementById("metricSelect");

const metricLabels = {
  risk_score: "Overall risk score",
  risk_new_cases_county: "Risk of new county cases",
  growth_pressure_score: "Growth pressure score",
  expected_growth_abs: "Expected growth by number"
};

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return await res.json();
}

function initMap() {
  map = L.map("map").setView([37.8, -96], 4);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 10,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);
}

function getCurrentRows() {
  const horizon = horizonSelect.value;
  return forecastByHorizon[horizon] || [];
}

function buildLookup(rows) {
  const out = {};
  rows.forEach(r => {
    out[String(r.county_fips).padStart(5, "0")] = r;
  });
  return out;
}

function getColorScaled(value, minVal, maxVal) {
  if (!Number.isFinite(value)) value = 0;
  if (!Number.isFinite(minVal)) minVal = 0;
  if (!Number.isFinite(maxVal)) maxVal = 1;

  if (maxVal <= minVal) return "#FFEDA0";

  const t = (value - minVal) / (maxVal - minVal);

  return t > 0.90 ? "#800026" :
         t > 0.75 ? "#BD0026" :
         t > 0.60 ? "#E31A1C" :
         t > 0.45 ? "#FC4E2A" :
         t > 0.30 ? "#FD8D3C" :
         t > 0.15 ? "#FEB24C" :
                    "#FFEDA0";
}

function renderLegend(metric, minVal, maxVal) {
  const legend = document.getElementById("legend");
  legend.innerHTML = `
    <strong>${metricLabels[metric]}</strong><br/>
    Dynamic scale for selected horizon.<br/>
    Min: ${Number.isFinite(minVal) ? minVal.toFixed(4) : "0.0000"} |
    Max: ${Number.isFinite(maxVal) ? maxVal.toFixed(4) : "0.0000"}
  `;
}

function renderMap() {
  const rows = getCurrentRows();
  const metric = metricSelect.value;

  let values = rows
    .map(r => Number(r[metric]))
    .filter(v => Number.isFinite(v));

  const minVal = values.length ? Math.min(...values) : 0;
  const maxVal = values.length ? Math.max(...values) : 1;

  if (geoLayer) {
    geoLayer.remove();
  }

  if (!countyGeojson || !countyGeojson.features || countyGeojson.features.length === 0) {
    renderLegend(metric, minVal, maxVal);
    return;
  }

  const lookup = buildLookup(rows);
  const geo = JSON.parse(JSON.stringify(countyGeojson));

  geo.features.forEach(f => {
    const fips = String(f.id).padStart(5, "0");
    const row = lookup[fips];
    f.properties = f.properties || {};

    if (row) {
      f.properties.metricValue = Number(row[metric] || 0);
      f.properties.county = row.county || f.properties.NAME || "";
      f.properties.state = row.state || "";
      f.properties.expected_growth_abs = Number(row.expected_growth_abs || 0);
      f.properties.expected_growth_pct = Number(row.expected_growth_pct || 0);
      f.properties.risk_new_cases_county = Number(row.risk_new_cases_county || 0);
      f.properties.new_cases_risk_pct = Number(row.new_cases_risk_pct || 0);
      f.properties.risk_score = Number(row.risk_score || 0);
      f.properties.risk_score_percentile = Number(row.risk_score_percentile || 0);
      f.properties.growth_pressure_score = Number(row.growth_pressure_score || 0);
    } else {
      f.properties.metricValue = 0;
      f.properties.county = f.properties.NAME || "";
      f.properties.state = "";
      f.properties.expected_growth_abs = 0;
      f.properties.expected_growth_pct = 0;
      f.properties.risk_new_cases_county = 0;
      f.properties.new_cases_risk_pct = 0;
      f.properties.risk_score = 0;
      f.properties.risk_score_percentile = 0;
      f.properties.growth_pressure_score = 0;
    }
  });

  geoLayer = L.geoJSON(geo, {
    style: feature => ({
      fillColor: getColorScaled(feature.properties.metricValue, minVal, maxVal),
      weight: 0.3,
      opacity: 1,
      color: "white",
      fillOpacity: 0.8
    }),
    onEachFeature: (feature, layer) => {
      const p = feature.properties;
      const html = `
        <div>
          <strong>${p.county || "County"}</strong><br/>
          State: ${p.state || ""}<br/>
          Overall risk score: ${Number(p.risk_score || 0).toFixed(4)}<br/>
          Overall risk percentile: ${Number(p.risk_score_percentile || 0).toFixed(1)}<br/>
          New county case risk: ${Number(p.new_cases_risk_pct || 0).toFixed(2)}%<br/>
          Growth pressure score: ${Number(p.growth_pressure_score || 0).toFixed(1)}<br/>
          Expected growth by number: ${Number(p.expected_growth_abs || 0).toFixed(4)}
        </div>
      `;
      layer.bindPopup(html);
    }
  }).addTo(map);

  renderLegend(metric, minVal, maxVal);
}

function fillTable(tableId, rows) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  tbody.innerHTML = "";

  rows.slice(0, 25).forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.county || ""}</td>
      <td>${r.state || ""}</td>
      <td>${Number(r.risk_score || 0).toFixed(4)}</td>
      <td>${Number(r.new_cases_risk_pct || 0).toFixed(2)}%</td>
      <td>${Number(r.growth_pressure_score || 0).toFixed(1)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderTables() {
  const horizon = horizonSelect.value;
  const data = tablesByHorizon[horizon];
  if (!data) return;

  fillTable("tableAbs", data.top_risk_score || []);
  fillTable("tablePct", data.top_new_area_risk || []);
  fillTable("tableNew", data.top_growth_pressure || []);
}

function rerender() {
  renderMap();
  renderTables();
}

async function boot() {
  initMap();

  forecastByHorizon = await loadJSON("./data/forecast_by_horizon.json");
  tablesByHorizon = await loadJSON("./data/tables.json");

  try {
    countyGeojson = await loadJSON("./data/counties_h1.geojson");
  } catch (err) {
    console.warn("Could not load counties_h1.geojson, continuing without county polygons.", err);
    countyGeojson = { type: "FeatureCollection", features: [] };
  }

  horizonSelect.addEventListener("change", rerender);
  metricSelect.addEventListener("change", rerender);

  rerender();
}

boot().catch(err => {
  console.error(err);
  alert("Failed to load website data. Check console.");
});