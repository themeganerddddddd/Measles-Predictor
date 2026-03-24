let map;
let geoLayer;
let forecastByHorizon = {};
let tablesByHorizon = {};
let countyGeojson = null;
let metadata = {};

const horizonSelect = document.getElementById("horizonSelect");
const metricSelect = document.getElementById("metricSelect");

const metricLabels = {
  risk_score: "Overall County Risk",
  new_cases_risk_pct: "Emergence Risk",
  growth_pressure_score: "Spread Pressure"
};

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to load ${path} (${res.status})`);
  }
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
  const horizon = String(horizonSelect.value);
  return forecastByHorizon[horizon] || [];
}

function buildLookup(rows) {
  const out = {};
  rows.forEach(r => {
    const fips = String(r.county_fips || "").padStart(5, "0");
    if (fips) out[fips] = r;
  });
  return out;
}

function safeNum(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function getColorScaled(value, minVal, maxVal) {
  value = safeNum(value, 0);
  minVal = safeNum(minVal, 0);
  maxVal = safeNum(maxVal, 1);

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
    <strong>${metricLabels[metric] || metric}</strong><br/>
    Dynamic scale for selected horizon.<br/>
    Min: ${Number.isFinite(minVal) ? minVal.toFixed(2) : "0.00"} |
    Max: ${Number.isFinite(maxVal) ? maxVal.toFixed(2) : "0.00"}
  `;
}

function updateMetadataUI() {
  const lastDataDateEl = document.getElementById("lastDataDate");
  const lastRefreshedEl = document.getElementById("lastRefreshed");
  const forecastWeekEl = document.getElementById("forecastWeek");

  if (lastDataDateEl) {
    lastDataDateEl.textContent = metadata.last_data_date || "—";
  }

  if (lastRefreshedEl) {
    lastRefreshedEl.textContent = metadata.last_refreshed || "—";
  }

  if (forecastWeekEl) {
    const horizon = String(horizonSelect.value);
    const forecastWeek = metadata.forecast_week_by_horizon?.[horizon] || "—";
    forecastWeekEl.textContent = forecastWeek;
  }
}

function renderMap() {
  const rows = getCurrentRows();
  const metric = metricSelect.value;

  const values = rows
    .map(r => safeNum(r[metric], NaN))
    .filter(v => Number.isFinite(v));

  const minVal = values.length ? Math.min(...values) : 0;
  const maxVal = values.length ? Math.max(...values) : 1;

  if (geoLayer) {
    geoLayer.remove();
    geoLayer = null;
  }

  if (!countyGeojson || !countyGeojson.features || countyGeojson.features.length === 0) {
    renderLegend(metric, minVal, maxVal);
    return;
  }

  const lookup = buildLookup(rows);
  const geo = JSON.parse(JSON.stringify(countyGeojson));

  geo.features.forEach(feature => {
    const fips = String(feature.id || "").padStart(5, "0");
    const row = lookup[fips];
    feature.properties = feature.properties || {};

    if (row) {
      feature.properties.metricValue = safeNum(row[metric], 0);
      feature.properties.county = row.county || feature.properties.NAME || "";
      feature.properties.state = row.state || "";
      feature.properties.risk_score = safeNum(row.risk_score, 0);
      feature.properties.risk_score_percentile = safeNum(row.risk_score_percentile, 0);
      feature.properties.new_cases_risk_pct = safeNum(row.new_cases_risk_pct, 0);
      feature.properties.growth_pressure_score = safeNum(row.growth_pressure_score, 0);
    } else {
      feature.properties.metricValue = 0;
      feature.properties.county = feature.properties.NAME || "";
      feature.properties.state = "";
      feature.properties.risk_score = 0;
      feature.properties.risk_score_percentile = 0;
      feature.properties.new_cases_risk_pct = 0;
      feature.properties.growth_pressure_score = 0;
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
          Overall County Risk: ${safeNum(p.risk_score).toFixed(2)}<br/>
          Overall Risk Percentile: ${safeNum(p.risk_score_percentile).toFixed(1)}<br/>
          Emergence Risk: ${safeNum(p.new_cases_risk_pct).toFixed(1)}<br/>
          Spread Pressure: ${safeNum(p.growth_pressure_score).toFixed(1)}
        </div>
      `;
      layer.bindPopup(html);
    }
  }).addTo(map);

  renderLegend(metric, minVal, maxVal);
}

function fillTable(tableId, rows) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  if (!tbody) return;

  tbody.innerHTML = "";

  rows.slice(0, 25).forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.county || ""}</td>
      <td>${r.state || ""}</td>
      <td>${safeNum(r.risk_score).toFixed(2)}</td>
      <td>${safeNum(r.new_cases_risk_pct).toFixed(1)}</td>
      <td>${safeNum(r.growth_pressure_score).toFixed(1)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderTables() {
  const horizon = String(horizonSelect.value);
  const data = tablesByHorizon[horizon];
  if (!data) return;

  fillTable("tableAbs", data.top_risk_score || []);
  fillTable("tablePct", data.top_new_area_risk || []);
  fillTable("tableNew", data.top_growth_pressure || []);
}

function rerender() {
  updateMetadataUI();
  renderMap();
  renderTables();
}

async function boot() {
  initMap();

  forecastByHorizon = await loadJSON("./data/forecast_by_horizon.json");
  tablesByHorizon = await loadJSON("./data/tables.json");
  metadata = await loadJSON("./data/metadata.json");

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