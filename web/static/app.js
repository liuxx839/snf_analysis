// ---------- tab switching ----------
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

let META = null;
let LAST_TRAIN = null;

async function fetchJSON(url, opts={}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `${res.status}`);
  }
  return res.json();
}

// ---------- build patient form + filters ----------
async function init() {
  META = await fetchJSON("/api/meta");

  const form = document.getElementById("patient-form");
  form.innerHTML = "";
  META.features.forEach(feat => {
    form.appendChild(buildField(feat));
  });

  const numFilters = document.getElementById("numeric-filters");
  const catFilters = document.getElementById("categorical-filters");
  numFilters.innerHTML = "";
  catFilters.innerHTML = "";
  META.numeric_features.forEach(f => numFilters.appendChild(buildNumericFilter(f)));
  META.categorical_features.forEach(f => catFilters.appendChild(buildCategoricalFilter(f)));

  const fbox = document.getElementById("feature-checkboxes");
  fbox.innerHTML = "";
  META.features.forEach(f => {
    const wrap = document.createElement("label");
    wrap.innerHTML = `<input type="checkbox" data-feature="${f}" checked> ${f}`;
    fbox.appendChild(wrap);
  });

  const savedDefault = localStorage.getItem("snf_patient_default");
  if (savedDefault) {
    try { fillPatientForm(JSON.parse(savedDefault)); } catch {}
  }
  updateFilterStat();
}

function buildField(name) {
  const meta = META.columns[name];
  const wrap = document.createElement("div");
  wrap.className = "field";
  const lab = document.createElement("label");
  lab.textContent = name;
  wrap.appendChild(lab);

  if (meta.type === "numeric") {
    const inp = document.createElement("input");
    inp.type = "number";
    inp.step = "any";
    inp.name = name;
    inp.placeholder = `中位数 ${meta.median?.toFixed(1) ?? ""}`;
    wrap.appendChild(inp);
    const s = document.createElement("small");
    s.textContent = `范围 ${meta.min?.toFixed(1)}–${meta.max?.toFixed(1)} · 缺失率 ${meta.missing_pct.toFixed(0)}%`;
    wrap.appendChild(s);
  } else {
    const sel = document.createElement("select");
    sel.name = name;
    const opt0 = document.createElement("option");
    opt0.value = ""; opt0.textContent = "(未知 / 不填)";
    sel.appendChild(opt0);
    meta.values.forEach((v, i) => {
      const o = document.createElement("option");
      o.value = v; o.textContent = `${v}  (n=${meta.counts[i]})`;
      sel.appendChild(o);
    });
    wrap.appendChild(sel);
  }
  return wrap;
}

function buildNumericFilter(name) {
  const meta = META.columns[name];
  const wrap = document.createElement("div");
  wrap.className = "field";
  const lab = document.createElement("label");
  lab.textContent = name;
  wrap.appendChild(lab);
  const row = document.createElement("div");
  row.className = "range-inputs";
  const mn = document.createElement("input");
  mn.type = "number"; mn.step = "any"; mn.placeholder = `≥ ${meta.min?.toFixed(1)}`;
  mn.dataset.filterCol = name; mn.dataset.filterSide = "min";
  mn.addEventListener("input", updateFilterStat);
  const mx = document.createElement("input");
  mx.type = "number"; mx.step = "any"; mx.placeholder = `≤ ${meta.max?.toFixed(1)}`;
  mx.dataset.filterCol = name; mx.dataset.filterSide = "max";
  mx.addEventListener("input", updateFilterStat);
  row.appendChild(mn); row.appendChild(mx);
  wrap.appendChild(row);
  return wrap;
}

function buildCategoricalFilter(name) {
  const meta = META.columns[name];
  const wrap = document.createElement("div");
  wrap.className = "field";
  const lab = document.createElement("label");
  lab.textContent = `${name}  (N_total=${meta.counts.reduce((a,b)=>a+b,0)})`;
  wrap.appendChild(lab);
  const inner = document.createElement("div");
  meta.values.forEach((v, i) => {
    const l = document.createElement("label");
    l.style.display = "inline-flex";
    l.style.marginRight = "10px";
    l.style.fontSize = "12px";
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.value = v;
    cb.dataset.filterCol = name;
    cb.addEventListener("change", updateFilterStat);
    l.appendChild(cb);
    l.appendChild(document.createTextNode(`${v} (${meta.counts[i]})`));
    inner.appendChild(l);
  });
  wrap.appendChild(inner);
  return wrap;
}

function collectFilters() {
  const filters = {};
  document.querySelectorAll("[data-filter-side]").forEach(inp => {
    if (inp.value === "" || inp.value === null) return;
    const col = inp.dataset.filterCol;
    filters[col] = filters[col] || {};
    filters[col][inp.dataset.filterSide] = parseFloat(inp.value);
  });
  const byCat = {};
  document.querySelectorAll("input[type=checkbox][data-filter-col]").forEach(cb => {
    const col = cb.dataset.filterCol;
    if (!cb.checked) return;
    byCat[col] = byCat[col] || [];
    byCat[col].push(cb.value);
  });
  Object.entries(byCat).forEach(([k, v]) => { if (v.length) filters[k] = v; });
  return filters;
}

function collectFeatures() {
  return [...document.querySelectorAll("#feature-checkboxes input:checked")]
    .map(cb => cb.dataset.feature);
}

function collectPatient() {
  const obj = { patient_id: "ME" };
  document.querySelectorAll("#patient-form [name]").forEach(el => {
    const v = el.value;
    if (v === "" || v === null) return;
    obj[el.name] = el.type === "number" ? Number(v) : v;
  });
  return obj;
}

function fillPatientForm(data) {
  document.querySelectorAll("#patient-form [name]").forEach(el => {
    if (data[el.name] !== undefined && data[el.name] !== null) {
      el.value = data[el.name];
    }
  });
}

function estimateSubpopulation() {
  return "(点击训练即可看到)";
}

function updateFilterStat() {
  document.getElementById("filter-stat").textContent =
    "筛选条件已更新,点击训练按钮在子人群上重新拟合模型。";
}

// ---------- predict ----------
document.getElementById("btn-predict").addEventListener("click", async () => {
  const patient = collectPatient();
  const btn = document.getElementById("btn-predict");
  btn.disabled = true; btn.textContent = "预测中...";
  try {
    const r = await fetchJSON("/api/predict", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify({ patient }),
    });
    renderPrediction(r);
  } catch (e) {
    alert("预测失败: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "预测 SNF 亚型";
  }
});

document.getElementById("btn-save-default").addEventListener("click", () => {
  localStorage.setItem("snf_patient_default", JSON.stringify(collectPatient()));
  alert("已保存为浏览器默认。");
});
document.getElementById("btn-load-default").addEventListener("click", () => {
  const s = localStorage.getItem("snf_patient_default");
  if (!s) { alert("没有已保存的默认值。"); return; }
  fillPatientForm(JSON.parse(s));
});
document.getElementById("btn-load-example").addEventListener("click", () => {
  fillPatientForm({
    Age: 45, Tumor_size_cm: 2.2, Positive_axillary_lymph_nodes: 0,
    ER_percent: 90, PR_percent: 80, Ki67: 20, HER2_IHC_Status: 1,
    Menopause: "No", Grade: "2", pT: "pT2", pN: "pN0", PR_status: "Positive",
  });
});

function renderPrediction(r) {
  const card = document.getElementById("predict-result");
  card.classList.remove("hidden");

  const cls = { high: "high", medium: "medium", low: "low" }[r.confidence];
  document.getElementById("predict-summary").innerHTML = `
    <p>最可能的亚型: <b style="font-size:18px; color: var(--brand);">${r.predicted_subtype}</b>
       <span class="badge ${cls}">置信度: ${{"high":"较高","medium":"中等","low":"偏低"}[r.confidence]}</span></p>
  `;

  const bars = document.getElementById("prob-bars");
  bars.innerHTML = "";
  const order = ["SNF1","SNF2","SNF3","SNF4"].filter(c => r.probabilities[c]);
  order.forEach(c => {
    const p = r.probabilities[c];
    const row = document.createElement("div");
    row.className = "prob-bar";
    row.innerHTML = `
      <div class="label">${c}</div>
      <div class="track">
        <div class="fill" style="width:${(p.prob*100).toFixed(1)}%"></div>
        <div class="ci" style="left:${(p.lo*100).toFixed(1)}%; width:${((p.hi-p.lo)*100).toFixed(1)}%"></div>
      </div>
      <div class="meta">${(p.prob*100).toFixed(1)}% &nbsp; [${(p.lo*100).toFixed(1)}–${(p.hi*100).toFixed(1)}]</div>
    `;
    bars.appendChild(row);
  });

  document.getElementById("interpretation").textContent = r.interpretation;

  const descBox = document.getElementById("subtype-desc");
  descBox.innerHTML = "";
  Object.entries(r.subtype_description).forEach(([k, v]) => {
    const p = document.createElement("p");
    p.innerHTML = `<b>${k}:</b> ${v}`;
    descBox.appendChild(p);
  });

  let info = "";
  if (r.model_performance) {
    const m = r.model_performance;
    info = `当前预测用的模型: N=${m.n_samples},Macro AUC = ${m.macro_auc.toFixed(3)} [${m.macro_auc_ci[0].toFixed(3)}, ${m.macro_auc_ci[1].toFixed(3)}]。`;
  } else {
    info = "当前使用默认全队列模型。到 Tab ② 训练自定义模型后,预测会切换到新模型。";
  }
  if (Object.keys(r.filters_applied || {}).length) {
    info += "  子人群条件: " + JSON.stringify(r.filters_applied);
  }
  document.getElementById("predict-model-info").textContent = info;
}

// ---------- train ----------
document.getElementById("btn-train").addEventListener("click", async () => {
  const payload = {
    features: collectFeatures(),
    filters: collectFilters(),
    n_splits: parseInt(document.getElementById("n-splits").value),
    n_boot: parseInt(document.getElementById("n-boot").value),
    n_estimators: parseInt(document.getElementById("n-estimators").value),
  };
  const btn = document.getElementById("btn-train");
  btn.disabled = true; btn.textContent = "训练中,请稍候...";
  try {
    const r = await fetchJSON("/api/train", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify(payload),
    });
    LAST_TRAIN = r;
    renderTrainResult(r);
    document.querySelector('.tab[data-tab="tab-results"]').click();
  } catch (e) {
    alert("训练失败: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "在子人群上训练";
  }
});

document.getElementById("btn-reset-filter").addEventListener("click", () => {
  document.querySelectorAll("#numeric-filters input").forEach(i => i.value = "");
  document.querySelectorAll("#categorical-filters input[type=checkbox]").forEach(i => i.checked = false);
  updateFilterStat();
});

async function renderTrainResult(r) {
  const bench = await fetchJSON("/api/benchmarks");

  const classCounts = Object.entries(r.class_counts).map(([k,v]) => `${k}=${v}`).join(", ");
  const filtStr = Object.keys(r.filters_applied||{}).length ? JSON.stringify(r.filters_applied) : "(全队列)";
  document.getElementById("cv-summary").innerHTML = `
    <p>训练样本 N = <b>${r.n_samples}</b> (${classCounts})</p>
    <p>子人群条件: <code>${escapeHTML(filtStr)}</code></p>
    <p>使用特征: <code>${r.features_used.join(", ")}</code></p>
    <p>Macro AUC = <b>${r.macro_auc.toFixed(3)}</b>
       95% CI [${r.macro_auc_ci[0].toFixed(3)}, ${r.macro_auc_ci[1].toFixed(3)}]
       &nbsp;·&nbsp; 逐折: ${r.fold_macro_auc.map(v=>v.toFixed(3)).join(", ")}</p>
  `;

  const benchBox = document.getElementById("bench-table");
  const rows = [];
  rows.push(
    `<div class="bench-row header">
       <div class="name">模型</div>
       <div class="val">SNF1</div><div class="val">SNF2</div>
       <div class="val">SNF3</div><div class="val">SNF4</div>
       <div class="val">Macro</div>
     </div>`
  );
  const mine = r.per_class_auc;
  const mineCI = r.per_class_auc_ci;
  const myCells = ["SNF1","SNF2","SNF3","SNF4"].map(c => {
    if (mine[c] == null) return `<div class="val me">-</div>`;
    const lo = mineCI[c][0].toFixed(2), hi = mineCI[c][1].toFixed(2);
    return `<div class="val me" title="95% CI ${lo}-${hi}">${mine[c].toFixed(2)}</div>`;
  }).join("");
  rows.push(
    `<div class="bench-row">
       <div class="name">本模型(你的)</div>${myCells}
       <div class="val me">${r.macro_auc.toFixed(2)}<br><small>[${r.macro_auc_ci[0].toFixed(2)}, ${r.macro_auc_ci[1].toFixed(2)}]</small></div>
     </div>`
  );
  Object.entries(bench.per_class).forEach(([name, d]) => {
    const cls = name.includes("CNN") ? "paper-cnn" : "paper-rf";
    const cells = ["SNF1","SNF2","SNF3","SNF4"].map(c =>
      `<div class="val ${cls}">${d[c]?.toFixed(2) ?? "-"}</div>`).join("");
    const macro = (Object.values(d).reduce((a,b)=>a+b,0)/Object.values(d).length).toFixed(2);
    rows.push(
      `<div class="bench-row">
         <div class="name">${name}(原文)</div>${cells}
         <div class="val ${cls}">${macro}</div>
       </div>`
    );
  });
  benchBox.innerHTML = rows.join("");

  drawROC(r, bench);

  const cm = r.confusion_matrix;
  const labels = r.labels;
  let html = '<table class="data cm-table"><thead><tr><th>真实\\预测</th>';
  labels.forEach(l => html += `<th>${l}</th>`);
  html += `<th>合计</th></tr></thead><tbody>`;
  cm.forEach((row, i) => {
    const total = row.reduce((a,b)=>a+b,0);
    html += `<tr><th>${labels[i]}</th>`;
    row.forEach((v, j) => {
      html += `<td class="${i===j?'diag':''}">${v}</td>`;
    });
    html += `<td>${total}</td></tr>`;
  });
  html += `</tbody></table>`;
  document.getElementById("cm-table").innerHTML = html;

  const fi = r.feature_importance_top15;
  const maxImp = fi[0]?.importance || 1;
  document.getElementById("feat-imp").innerHTML = fi.map(f => `
    <div class="imp-bar">
      <div><code>${f.name}</code></div>
      <div class="track"><div class="fill" style="width:${(100*f.importance/maxImp).toFixed(1)}%"></div></div>
    </div>
    <div style="font-size:11px; color:var(--muted); margin-left:4px; margin-bottom:3px;">imp = ${f.importance.toFixed(4)}</div>
  `).join("");

  document.getElementById("classification-report").textContent = r.classification_report;
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

// ---------- draw ROC on canvas ----------
function drawROC(r, bench) {
  const canvas = document.getElementById("roc-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const P = 45;
  ctx.clearRect(0,0,W,H);

  ctx.strokeStyle = "#cbd5e1"; ctx.fillStyle = "#475569"; ctx.font = "12px sans-serif";
  ctx.strokeRect(P, P, W-2*P, H-2*P);
  for (let t=0; t<=10; t++) {
    const x = P + t*(W-2*P)/10, y = P + t*(H-2*P)/10;
    ctx.strokeStyle = "#eef2f7";
    ctx.beginPath(); ctx.moveTo(x, P); ctx.lineTo(x, H-P); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(P, y); ctx.lineTo(W-P, y); ctx.stroke();
  }
  ctx.strokeStyle = "#94a3b8";
  ctx.beginPath(); ctx.moveTo(P, H-P); ctx.lineTo(W-P, P); ctx.stroke();

  ctx.fillStyle = "#1f2937";
  ctx.fillText("False Positive Rate", W/2-50, H-10);
  ctx.save(); ctx.translate(14, H/2+50); ctx.rotate(-Math.PI/2);
  ctx.fillText("True Positive Rate", 0, 0); ctx.restore();

  const colors = { SNF1:"#2563eb", SNF2:"#059669", SNF3:"#d97706", SNF4:"#dc2626" };

  const labels = r.labels;
  labels.forEach(c => {
    const d = r.roc_points[c];
    if (!d) return;
    ctx.strokeStyle = colors[c] || "#333";
    ctx.lineWidth = 2;
    ctx.beginPath();
    d.fpr.forEach((x, i) => {
      const xp = P + x*(W-2*P), yp = H-P - d.tpr[i]*(H-2*P);
      if (i===0) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
    });
    ctx.stroke();
  });

  let ly = P + 8;
  ctx.font = "12px sans-serif";
  labels.forEach(c => {
    ctx.fillStyle = colors[c] || "#333";
    ctx.fillRect(W-P-160, ly, 14, 10);
    ctx.fillStyle = "#1f2937";
    const ci = r.per_class_auc_ci[c];
    ctx.fillText(`${c}  AUC ${r.per_class_auc[c].toFixed(2)} [${ci[0].toFixed(2)}, ${ci[1].toFixed(2)}]`,
                 W-P-144, ly+9);
    ly += 16;
  });

  ly += 8;
  ctx.fillStyle = "#475569"; ctx.fillText("原文参考:", W-P-160, ly);
  ly += 14;
  Object.entries(bench.per_class).forEach(([name, d]) => {
    const macro = (Object.values(d).reduce((a,b)=>a+b,0)/Object.values(d).length);
    ctx.fillStyle = name.includes("CNN") ? "#b45309" : "#166534";
    ctx.fillText(`${name}: macro ≈ ${macro.toFixed(2)}`, W-P-160, ly);
    ly += 14;
  });
}

// ---------- similar ----------
document.getElementById("btn-similar").addEventListener("click", async () => {
  const payload = {
    patient: collectPatient(),
    k: parseInt(document.getElementById("sim-k").value),
    same_subtype_only: document.getElementById("sim-same").checked,
    weight_by_importance: document.getElementById("sim-weight").checked,
  };
  const btn = document.getElementById("btn-similar");
  btn.disabled = true; btn.textContent = "查找中...";
  try {
    const r = await fetchJSON("/api/similar", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify(payload),
    });
    renderSimilar(r);
  } catch (e) {
    alert("失败: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "查找";
  }
});

function renderSimilar(r) {
  const sum = document.getElementById("sim-summary");
  const dist = Object.entries(r.subtype_distribution).map(([k,v]) => `${k}=${v}`).join(", ");
  const svg = [];
  for (const key of ["OS","RFS","DMFS"]) {
    const s = r.survival_summary[key];
    if (s) svg.push(`${key}: 随访 ${s.n} 人, 事件 ${s.events} 人, 中位 ${s.median_months.toFixed(1)} 月`);
  }
  sum.innerHTML = `
    <p>预测亚型: <b>${r.predicted_subtype}</b> &nbsp;·&nbsp; 返回 ${r.k} 人 &nbsp;·&nbsp;
       亚型分布: ${dist}</p>
    <p class="hint">${svg.join(" &nbsp;|&nbsp; ")}</p>
  `;

  const cols = r.columns;
  let html = '<table class="data"><thead><tr>';
  cols.forEach(c => html += `<th>${c}</th>`);
  html += "</tr></thead><tbody>";
  r.rows.forEach(row => {
    html += "<tr>";
    cols.forEach(c => {
      let v = row[c];
      if (typeof v === "number") v = Number.isInteger(v) ? v : v.toFixed(2);
      if (v === null || v === undefined) v = "";
      html += `<td>${v}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  document.getElementById("sim-table").innerHTML = html;
}

init();
