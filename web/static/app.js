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
    const isTreatment = (META.treatment_features || []).includes(f);
    const desc = META.feature_description ? META.feature_description[f] : "";
    const tag = isTreatment ? ' <span class="badge medium" style="font-size:10px">治疗端·默认不勾</span>' : "";
    const checked = isTreatment ? "" : "checked";
    wrap.innerHTML = `<input type="checkbox" data-feature="${f}" ${checked}> ${f}${tag}`;
    wrap.title = desc || f;
    fbox.appendChild(wrap);
  });

  const modelSelect = document.getElementById("train-model");
  modelSelect.innerHTML = "";
  (META.available_models || []).forEach(m => {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    if (m === "RandomForest") o.selected = true;
    modelSelect.appendChild(o);
  });

  const cmpBox = document.getElementById("compare-models-box");
  cmpBox.innerHTML = "";
  (META.available_models || []).forEach(m => {
    const l = document.createElement("label");
    l.innerHTML = `<input type="checkbox" data-cmp-model="${m}" checked> ${m}`;
    cmpBox.appendChild(l);
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
  const desc = META.feature_description ? META.feature_description[name] : "";
  const isTreatment = (META.treatment_features || []).includes(name);
  lab.innerHTML = isTreatment
    ? `${name} <span class="badge medium" style="font-size:10px">治疗端</span>`
    : name;
  if (desc) lab.title = desc;
  wrap.appendChild(lab);
  if (desc) {
    const h = document.createElement("small");
    h.textContent = desc;
    h.style.color = "var(--muted)";
    wrap.appendChild(h);
  }

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

  let interp = r.interpretation;
  const isBagging = ["RandomForest","ExtraTrees"].includes(r.model_name);
  if (!isBagging) {
    interp += ` 注:当前算法为 ${r.model_name},没有"树集成方差"这种天然的预测不确定度,CI 退化为点估计(没有区间宽度);AUC 的 CI 仍由 bootstrap 给出(见 Tab ④)。`;
  }
  document.getElementById("interpretation").textContent = interp;

  const descBox = document.getElementById("subtype-desc");
  descBox.innerHTML = "";
  Object.entries(r.subtype_description).forEach(([k, v]) => {
    const p = document.createElement("p");
    p.innerHTML = `<b>${k}:</b> ${v}`;
    descBox.appendChild(p);
  });

  let info = "";
  const algo = r.model_name ? ` [${r.model_name}]` : "";
  if (r.model_performance) {
    const m = r.model_performance;
    info = `当前预测用的模型${algo}: N=${m.n_samples},Macro AUC = ${m.macro_auc.toFixed(3)} [${m.macro_auc_ci[0].toFixed(3)}, ${m.macro_auc_ci[1].toFixed(3)}]。`;
  } else {
    info = `当前使用默认全队列模型${algo}。到 Tab ② 训练,或 Tab ③ 大比拼自动选最佳。`;
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
    model_name: document.getElementById("train-model").value,
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
    <p>算法: <b>${r.model_name || "RandomForest"}</b> &nbsp;·&nbsp; 训练样本 N = <b>${r.n_samples}</b> (${classCounts})</p>
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
  if (!fi || fi.length === 0) {
    document.getElementById("feat-imp").innerHTML = `<p class="hint">该算法没有原生的特征重要性 / 系数可直接取用。</p>`;
  } else {
    const maxImp = fi[0].importance || 1;
    document.getElementById("feat-imp").innerHTML = fi.map(f => `
      <div class="imp-bar">
        <div><code>${f.name}</code></div>
        <div class="track"><div class="fill" style="width:${(100*f.importance/maxImp).toFixed(1)}%"></div></div>
      </div>
      <div style="font-size:11px; color:var(--muted); margin-left:4px; margin-bottom:3px;">imp = ${f.importance.toFixed(4)}</div>
    `).join("");
  }

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

// ---------- compare ----------
document.getElementById("btn-compare").addEventListener("click", async () => {
  const selModels = [...document.querySelectorAll("[data-cmp-model]:checked")]
    .map(cb => cb.dataset.cmpModel);
  if (!selModels.length) { alert("至少勾选一个模型"); return; }
  const payload = {
    features: collectFeatures(),
    filters: collectFilters(),
    model_names: selModels,
    n_splits: parseInt(document.getElementById("cmp-splits").value),
    n_boot: parseInt(document.getElementById("cmp-boot").value),
    n_estimators: parseInt(document.getElementById("cmp-trees").value),
    auto_select: document.getElementById("compare-auto").checked,
  };
  const btn = document.getElementById("btn-compare");
  const status = document.getElementById("compare-status");
  btn.disabled = true; btn.textContent = `跑 ${selModels.length} 个模型中...`;
  status.textContent = "可能需要 20-60 秒。";
  try {
    const r = await fetchJSON("/api/compare", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify(payload),
    });
    renderCompare(r);
    if (r.auto_selected_as_current) {
      status.textContent = `完成。当前 Tab ①④ 已切换为最佳模型: ${r.best_model}。`;
    } else {
      status.textContent = `完成。`;
    }
  } catch (e) {
    alert("比较失败: " + e.message);
    status.textContent = "";
  } finally {
    btn.disabled = false; btn.textContent = "开始比较";
  }
});

function renderCompare(r) {
  document.getElementById("compare-result").classList.remove("hidden");

  const rows = r.results;
  let html = `<table class="data"><thead><tr>
    <th>#</th><th>模型</th><th>Macro AUC</th><th>95% CI</th>
    <th>SNF1</th><th>SNF2</th><th>SNF3</th><th>SNF4</th>
    <th>N</th><th>耗时 (s)</th><th>状态</th>
  </tr></thead><tbody>`;
  rows.forEach((r, i) => {
    const isBest = (r.name === r && false) || i === 0 && r.fit_ok;
    if (!r.fit_ok) {
      html += `<tr><td>${i+1}</td><td>${r.name}</td><td colspan="8" style="color:#b91c1c">${escapeHTML(r.error || "FAILED")}</td><td>✗</td></tr>`;
      return;
    }
    const ci = `[${r.macro_auc_ci[0].toFixed(2)}, ${r.macro_auc_ci[1].toFixed(2)}]`;
    const cells = ["SNF1","SNF2","SNF3","SNF4"].map(c => {
      const a = r.per_class_auc[c];
      if (a == null) return "<td>-</td>";
      const cc = r.per_class_auc_ci[c];
      return `<td title="95% CI ${cc[0].toFixed(2)}–${cc[1].toFixed(2)}">${a.toFixed(2)}</td>`;
    }).join("");
    html += `<tr${i===0?' style="background:#eff6ff;font-weight:600"':''}>
       <td>${i+1}</td>
       <td>${r.name}${i===0?' 👑':''}</td>
       <td>${r.macro_auc.toFixed(3)}</td>
       <td>${ci}</td>
       ${cells}
       <td>${r.n_samples}</td>
       <td>${r.seconds}</td>
       <td>✓</td>
    </tr>`;
  });
  html += "</tbody></table>";
  document.getElementById("compare-table").innerHTML = html;

  drawCompareBars(rows, r.paper_benchmarks);
}

function drawCompareBars(rows, paper) {
  const canvas = document.getElementById("cmp-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const P = { top: 30, right: 20, bottom: 110, left: 58 };
  const okRows = rows.filter(r => r.fit_ok);

  const paperRows = Object.entries(paper).map(([name, d]) => {
    const vals = Object.values(d);
    const macro = vals.reduce((a,b)=>a+b,0) / vals.length;
    return { name: name + " (paper)", macro_auc: macro, macro_auc_ci: [macro, macro],
             is_paper: true };
  });
  const allRows = [...okRows, ...paperRows];

  const n = allRows.length;
  const innerW = W - P.left - P.right;
  const innerH = H - P.top - P.bottom;
  const barW = innerW / n * 0.72;
  const gap  = innerW / n * 0.28;

  ctx.strokeStyle = "#94a3b8"; ctx.fillStyle = "#6b7280"; ctx.font = "11px sans-serif";
  for (let t = 0; t <= 10; t++) {
    const y = P.top + innerH - t/10 * innerH;
    ctx.strokeStyle = "#eef2f7";
    ctx.beginPath(); ctx.moveTo(P.left, y); ctx.lineTo(P.left + innerW, y); ctx.stroke();
    ctx.fillStyle = "#6b7280";
    ctx.fillText((t/10).toFixed(1), 8, y+4);
  }
  ctx.strokeStyle = "#111"; ctx.beginPath();
  ctx.moveTo(P.left, P.top); ctx.lineTo(P.left, P.top + innerH);
  ctx.lineTo(P.left + innerW, P.top + innerH); ctx.stroke();

  ctx.save(); ctx.translate(14, P.top + innerH/2 + 40); ctx.rotate(-Math.PI/2);
  ctx.fillStyle = "#1f2937"; ctx.fillText("Macro AUC", 0, 0); ctx.restore();

  allRows.forEach((r, i) => {
    const x = P.left + (innerW / n) * i + gap/2;
    const y = P.top + innerH - r.macro_auc * innerH;
    const h = r.macro_auc * innerH;
    const color = r.is_paper
      ? (r.name.includes("CNN") ? "#fbbf24" : "#34d399")
      : (i === 0 ? "#2563eb" : "#93c5fd");
    ctx.fillStyle = color;
    ctx.fillRect(x, y, barW, h);

    if (!r.is_paper && r.macro_auc_ci) {
      const yLo = P.top + innerH - r.macro_auc_ci[0] * innerH;
      const yHi = P.top + innerH - r.macro_auc_ci[1] * innerH;
      const cx = x + barW/2;
      ctx.strokeStyle = "#1f2937"; ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(cx, yLo); ctx.lineTo(cx, yHi);
      ctx.moveTo(cx-5, yLo); ctx.lineTo(cx+5, yLo);
      ctx.moveTo(cx-5, yHi); ctx.lineTo(cx+5, yHi);
      ctx.stroke();
    }

    ctx.save();
    ctx.translate(x + barW/2, P.top + innerH + 6);
    ctx.rotate(Math.PI/3);
    ctx.fillStyle = "#111";
    ctx.font = "11px sans-serif";
    ctx.fillText(r.name, 0, 0);
    ctx.restore();

    ctx.fillStyle = "#0f172a";
    ctx.font = "10px sans-serif";
    ctx.fillText(r.macro_auc.toFixed(2), x + barW/2 - 11, y - 3);
  });
  ctx.font = "11px sans-serif"; ctx.fillStyle = "#374151";
  ctx.fillText("蓝色=本次训练, 橙/绿=原文基准, 误差线=95% bootstrap CI", P.left, 16);
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

// ---------- survival ----------
document.getElementById("btn-surv-train").addEventListener("click", async () => {
  const payload = {
    with_treatment: document.getElementById("surv-treatment").checked,
    n_splits: parseInt(document.getElementById("surv-splits").value),
    penalizer: parseFloat(document.getElementById("surv-pen").value),
  };
  const btn = document.getElementById("btn-surv-train");
  btn.disabled = true; btn.textContent = "训练中...";
  const status = document.getElementById("surv-status");
  status.textContent = "拟合 OS / RFS / DMFS 三个 Cox 模型, 约 5-15 秒...";
  try {
    const r = await fetchJSON("/api/survival/train", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify(payload),
    });
    renderSurvPerf(r);
    status.textContent = "训练完成, 可以点击「用当前病人预测」。";
  } catch (e) {
    alert("训练失败: " + e.message);
    status.textContent = "";
  } finally {
    btn.disabled = false; btn.textContent = "训练生存模型(OS + RFS + DMFS)";
  }
});

document.getElementById("btn-surv-predict").addEventListener("click", async () => {
  const patient = collectPatient();
  const btn = document.getElementById("btn-surv-predict");
  btn.disabled = true; btn.textContent = "预测中...";
  try {
    const r = await fetchJSON("/api/survival/predict", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify({ patient }),
    });
    renderSurvPrediction(r);
  } catch (e) {
    alert("预测失败: " + e.message + "\n如果还没训练模型, 请先点击「训练生存模型」。");
  } finally {
    btn.disabled = false; btn.textContent = "用当前病人预测";
  }
});

function renderSurvPerf(r) {
  document.getElementById("surv-perf").classList.remove("hidden");
  const box = document.getElementById("surv-perf-table");
  let html = `<table class="data"><thead><tr>
    <th>端点</th><th>N</th><th>事件数</th>
    <th>CV C-index (mean ± CI)</th><th>Train C-index</th><th>Test C-index</th>
  </tr></thead><tbody>`;
  Object.entries(r.endpoints).forEach(([ep, v]) => {
    if (v.error) {
      html += `<tr><td>${ep}</td><td colspan="5" style="color:#b91c1c">${escapeHTML(v.error)}</td></tr>`;
      return;
    }
    const ci = `[${v.cv_c_index_ci[0].toFixed(3)}, ${v.cv_c_index_ci[1].toFixed(3)}]`;
    html += `<tr>
      <td><b>${ep}</b></td><td>${v.n_total}</td><td>${v.n_events}</td>
      <td>${v.cv_c_index.toFixed(3)} &nbsp; ${ci}</td>
      <td>${v.train_c_index.toFixed(3)}</td>
      <td><b>${v.test_c_index.toFixed(3)}</b></td>
    </tr>`;
  });
  html += "</tbody></table><p class='hint'>C-index 约等于 AUC:0.5=随机,1.0=完美区分风险高低。在只用临床特征的前提下,DMFS/RFS 能到 0.75-0.80 已经很不错。</p>";
  box.innerHTML = html;
}

function renderSurvPrediction(r) {
  document.getElementById("surv-plot-wrap").classList.remove("hidden");
  drawSurvCurves(r.endpoints);

  const msBox = document.getElementById("surv-milestones");
  let html = `<h3>关键时间点生存概率</h3><table class="data"><thead><tr>
    <th>端点</th><th>Partial HR</th><th>2 年 (24 mo)</th><th>5 年 (60 mo)</th><th>10 年 (120 mo)</th><th>中位生存</th><th>Test C-index</th>
  </tr></thead><tbody>`;
  Object.entries(r.endpoints).forEach(([ep, v]) => {
    if (v.error) return;
    const p = v.prediction;
    const hr = p.partial_hazard;
    const hrTag = hr < 0.85 ? "color:#166534" : (hr > 1.15 ? "color:#b91c1c" : "");
    const med = p.median_survival_months == null ? "未达到" : `${p.median_survival_months.toFixed(0)} mo`;
    html += `<tr>
      <td><b>${ep}</b> ${escapeHTML(v.label)}</td>
      <td style="${hrTag}">${hr.toFixed(2)} ${hr<1?"(优于基线)":(hr>1?"(高于基线)":"")}</td>
      <td>${(p.milestones.p_survive_24mo*100).toFixed(1)}%</td>
      <td>${(p.milestones.p_survive_60mo*100).toFixed(1)}%</td>
      <td>${(p.milestones.p_survive_120mo*100).toFixed(1)}%</td>
      <td>${med}</td>
      <td>${v.model_performance.test_c_index.toFixed(3)}</td>
    </tr>`;
  });
  html += "</tbody></table>";
  msBox.innerHTML = html;

  const coefBox = document.getElementById("surv-coef");
  const firstOk = Object.entries(r.endpoints).find(([_, v]) => !v.error);
  if (firstOk) {
    const [ep, v] = firstOk;
    coefBox.innerHTML = `<b>${ep} 模型里最显著的前 8 个特征</b> (按 p 值排序, HR > 1 = 风险增加):<br>`
      + v.top_coefficients.map(c =>
          `<code>${c.feature}</code>: HR = ${c.hazard_ratio.toFixed(2)}, p = ${c.p.toExponential(2)}`
        ).join(" &nbsp;|&nbsp; ");
  }
}

function drawSurvCurves(endpoints) {
  const canvas = document.getElementById("surv-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const P = { top: 30, right: 20, bottom: 50, left: 60 };
  const innerW = W - P.left - P.right, innerH = H - P.top - P.bottom;

  // x: 0..120 mo, y: 0..1
  const maxT = 150;
  ctx.strokeStyle = "#cbd5e1"; ctx.fillStyle = "#6b7280"; ctx.font = "11px sans-serif";
  for (let t = 0; t <= 10; t++) {
    const y = P.top + innerH - t/10 * innerH;
    ctx.strokeStyle = "#eef2f7";
    ctx.beginPath(); ctx.moveTo(P.left, y); ctx.lineTo(P.left + innerW, y); ctx.stroke();
    ctx.fillStyle = "#6b7280"; ctx.fillText((t/10).toFixed(1), 22, y+4);
  }
  for (let k = 0; k <= 5; k++) {
    const x = P.left + innerW * k / 5;
    ctx.strokeStyle = "#eef2f7";
    ctx.beginPath(); ctx.moveTo(x, P.top); ctx.lineTo(x, P.top+innerH); ctx.stroke();
    ctx.fillStyle = "#6b7280"; ctx.fillText(`${Math.round(k*maxT/5)}mo`, x-14, P.top+innerH+14);
  }
  ctx.strokeStyle = "#111"; ctx.beginPath();
  ctx.moveTo(P.left, P.top); ctx.lineTo(P.left, P.top+innerH);
  ctx.lineTo(P.left+innerW, P.top+innerH); ctx.stroke();
  ctx.save(); ctx.translate(16, P.top + innerH/2 + 50); ctx.rotate(-Math.PI/2);
  ctx.fillStyle = "#111"; ctx.fillText("Survival probability", 0, 0); ctx.restore();
  ctx.fillText("Time (months)", P.left + innerW/2 - 30, H - 12);

  const colors = { OS:"#2563eb", RFS:"#059669", DMFS:"#d97706" };
  let legendY = P.top + 8;
  Object.entries(endpoints).forEach(([ep, v]) => {
    if (v.error) return;
    const p = v.prediction;
    ctx.strokeStyle = colors[ep] || "#333"; ctx.lineWidth = 2;
    ctx.beginPath();
    p.times.forEach((t, i) => {
      const x = P.left + Math.min(t, maxT) / maxT * innerW;
      const y = P.top + innerH - p.survival[i] * innerH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = colors[ep];
    ctx.fillRect(P.left + 10, legendY, 14, 10);
    ctx.fillStyle = "#111";
    ctx.fillText(`${ep} (test C = ${v.model_performance.test_c_index.toFixed(2)})`,
                 P.left + 30, legendY + 9);
    legendY += 16;
  });
}

init();
