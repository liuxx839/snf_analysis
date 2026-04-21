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
    const w = m.weighted_auc ?? m.macro_auc;
    const wci = m.weighted_auc_ci ?? m.macro_auc_ci;
    info = `当前预测用的模型${algo}: N=${m.n_samples}, Weighted AUC = ${w.toFixed(3)} [${wci[0].toFixed(3)}, ${wci[1].toFixed(3)}] (Macro = ${m.macro_auc.toFixed(3)})。`;
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
  const wauc = r.weighted_auc ?? r.macro_auc;
  const wci = r.weighted_auc_ci ?? r.macro_auc_ci;
  const foldW = r.fold_weighted_auc || [];
  const foldM = r.fold_macro_auc || [];
  const _mean = a => a.length ? (a.reduce((s,v)=>s+v,0)/a.length) : NaN;
  document.getElementById("cv-summary").innerHTML = `
    <p>算法: <b>${r.model_name || "RandomForest"}</b> &nbsp;·&nbsp; 训练样本 N = <b>${r.n_samples}</b> (${classCounts})</p>
    <p>子人群条件: <code>${escapeHTML(filtStr)}</code></p>
    <p>使用特征: <code>${r.features_used.join(", ")}</code></p>
    <p><b>Weighted AUC</b>(pooled OOF) = <b>${wauc.toFixed(3)}</b>
       95% CI [${wci[0].toFixed(3)}, ${wci[1].toFixed(3)}]
       &nbsp;·&nbsp; Macro AUC = ${r.macro_auc.toFixed(3)}</p>
    <p class="hint">
      <b>逐折 5 个 AUC</b>(每折单独计算,仅作参考):<br>
      weighted = ${foldW.map(v=>v.toFixed(3)).join(", ") || "n/a"}
      ${foldW.length ? `(mean = ${_mean(foldW).toFixed(3)})` : ""}<br>
      macro = ${foldM.map(v=>v.toFixed(3)).join(", ")}
      ${foldM.length ? `(mean = ${_mean(foldM).toFixed(3)})` : ""}<br>
      上面 <b>Weighted AUC = ${wauc.toFixed(3)}</b> 不是这 5 折的简单平均,而是把 5 折全部 out-of-fold 预测拼起来一次性算的(pooled OOF),数值更稳。
    </p>
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
  let html = `<p class="hint">按 <b>Weighted AUC</b> 排序(按每类样本数加权,小类权重小; 类别不均衡场景更有代表性)。
    同时显示 Macro AUC 供参考。</p>
    <table class="data"><thead><tr>
    <th>#</th><th>模型</th>
    <th>Weighted AUC</th><th>95% CI</th>
    <th>Macro AUC</th>
    <th>SNF1</th><th>SNF2</th><th>SNF3</th><th>SNF4</th>
    <th>N</th><th>耗时 (s)</th>
  </tr></thead><tbody>`;
  rows.forEach((r, i) => {
    if (!r.fit_ok) {
      html += `<tr><td>${i+1}</td><td>${r.name}</td><td colspan="9" style="color:#b91c1c">${escapeHTML(r.error || "FAILED")}</td></tr>`;
      return;
    }
    const wci = `[${r.weighted_auc_ci[0].toFixed(2)}, ${r.weighted_auc_ci[1].toFixed(2)}]`;
    const cells = ["SNF1","SNF2","SNF3","SNF4"].map(c => {
      const a = r.per_class_auc[c];
      if (a == null) return "<td>-</td>";
      const cc = r.per_class_auc_ci[c];
      return `<td title="95% CI ${cc[0].toFixed(2)}–${cc[1].toFixed(2)}">${a.toFixed(2)}</td>`;
    }).join("");
    html += `<tr${i===0?' style="background:#eff6ff;font-weight:600"':''}>
       <td>${i+1}</td>
       <td>${r.name}${i===0?' 👑':''}</td>
       <td>${r.weighted_auc.toFixed(3)}</td>
       <td>${wci}</td>
       <td>${r.macro_auc.toFixed(3)}</td>
       ${cells}
       <td>${r.n_samples}</td>
       <td>${r.seconds}</td>
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
  const okRows = rows.filter(r => r.fit_ok).map(r => ({
    ...r,
    auc: r.weighted_auc,
    auc_ci: r.weighted_auc_ci,
  }));

  const paperRows = Object.entries(paper).map(([name, d]) => {
    const vals = Object.values(d);
    const macro = vals.reduce((a,b)=>a+b,0) / vals.length;
    return { name: name + " (paper)", auc: macro, auc_ci: [macro, macro],
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
    const y = P.top + innerH - r.auc * innerH;
    const h = r.auc * innerH;
    const color = r.is_paper
      ? (r.name.includes("CNN") ? "#fbbf24" : "#34d399")
      : (i === 0 ? "#2563eb" : "#93c5fd");
    ctx.fillStyle = color;
    ctx.fillRect(x, y, barW, h);

    if (!r.is_paper && r.auc_ci) {
      const yLo = P.top + innerH - r.auc_ci[0] * innerH;
      const yHi = P.top + innerH - r.auc_ci[1] * innerH;
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
    ctx.fillText(r.auc.toFixed(2), x + barW/2 - 11, y - 3);
  });
  ctx.font = "11px sans-serif"; ctx.fillStyle = "#374151";
  ctx.fillText("y = Weighted AUC · 蓝色=本次训练, 橙/绿=原文基准(macro AUC), 误差线=95% bootstrap CI", P.left, 16);
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
let SURV_LAST_TRAIN = null;
let SURV_LAST_PREDICT = null;
let SURV_SELECTED_VARIANTS = new Set(["snf+treat"]);  // 默认只显示最全信息的那条

document.getElementById("btn-surv-train").addEventListener("click", async () => {
  const payload = {
    n_splits: parseInt(document.getElementById("surv-splits").value),
    penalizer: parseFloat(document.getElementById("surv-pen").value),
  };
  const btn = document.getElementById("btn-surv-train");
  btn.disabled = true; btn.textContent = "训练中...";
  const status = document.getElementById("surv-status");
  status.textContent = "拟合 12 个 Cox 模型(3 端点 × 4 变体), 约 10-30 秒...";
  try {
    const r = await fetchJSON("/api/survival/train", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify(payload),
    });
    SURV_LAST_TRAIN = r;
    renderSurvPerf(r);
    status.textContent = "训练完成 (Full + Matched 两组), 可以点击「用当前病人预测」。";
  } catch (e) {
    alert("训练失败: " + e.message);
    status.textContent = "";
  } finally {
    btn.disabled = false; btn.textContent = "训练 4 变体(OS+RFS+DMFS = 12 个模型)";
  }
});

async function survPredict() {
  const patient = collectPatient();
  const cohort = document.getElementById("surv-cohort").value;
  const r = await fetchJSON("/api/survival/predict", {
    method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify({ patient, cohort }),
  });
  renderSurvPrediction(r);
}

document.getElementById("btn-surv-predict").addEventListener("click", async () => {
  const btn = document.getElementById("btn-surv-predict");
  btn.disabled = true; btn.textContent = "预测中...";
  try {
    await survPredict();
  } catch (e) {
    alert("预测失败: " + e.message + "\n如果还没训练模型, 请先点击「训练生存模型」。");
  } finally {
    btn.disabled = false; btn.textContent = "用当前病人预测";
  }
});

// 切换 cohort/baseline 时自动重新拉一次或重画
document.getElementById("surv-cohort").addEventListener("change", async () => {
  if (!SURV_LAST_PREDICT) return;
  try { await survPredict(); } catch (e) { /* ignore */ }
});
document.getElementById("surv-show-baseline").addEventListener("change", () => {
  if (SURV_LAST_PREDICT) redrawSurvPrediction(SURV_LAST_PREDICT);
});

function renderSurvPerf(r) {
  document.getElementById("surv-perf").classList.remove("hidden");
  const vorder = r.variants_order || ["base","treat","snf","snf+treat"];
  const vmeta = r.variants_meta || {};
  const cohorts = r.cohorts || {
    full: { endpoints: r.endpoints, description: "" },
  };
  const _renderTable = (cohortData) => {
    let html = `<table class="data"><thead><tr><th rowspan="2">端点</th>`;
    vorder.forEach(vk => {
      const m = vmeta[vk] || {};
      html += `<th colspan="3" style="border-left:2px solid var(--border)">${escapeHTML(m.label || vk)}</th>`;
    });
    html += `</tr><tr>`;
    vorder.forEach(() => {
      html += `<th style="border-left:2px solid var(--border)">Train C</th><th>Test C</th><th>CV C [95% CI]</th>`;
    });
    html += `</tr></thead><tbody>`;
    Object.entries(cohortData.endpoints || {}).forEach(([ep, epData]) => {
      html += `<tr><td><b>${ep}</b><br><small>${escapeHTML(epData.label||"")}</small></td>`;
      vorder.forEach(vk => {
        const v = epData.variants[vk];
        if (!v || v.error) {
          html += `<td colspan="3" style="color:#b91c1c;border-left:2px solid var(--border)">ERR</td>`;
          return;
        }
        const ci = `[${v.cv_c_index_ci[0].toFixed(2)}, ${v.cv_c_index_ci[1].toFixed(2)}]`;
        html += `<td style="border-left:2px solid var(--border)">${v.train_c_index.toFixed(3)}</td>`;
        html += `<td><b>${v.test_c_index.toFixed(3)}</b><br><small>n=${v.n_total} ev=${v.n_events}</small></td>`;
        html += `<td>${v.cv_c_index.toFixed(3)}<br><small>${ci}</small></td>`;
      });
      html += `</tr>`;
    });
    html += "</tbody></table>";
    return html;
  };
  document.getElementById("surv-perf-table").innerHTML = _renderTable(cohorts.full || {});
  document.getElementById("surv-perf-matched").innerHTML = _renderTable(cohorts.matched || {endpoints:{}});
}

function renderSurvPrediction(r) {
  SURV_LAST_PREDICT = r;
  document.getElementById("surv-plot-wrap").classList.remove("hidden");

  // 顶部:展示 Tab ① 预测的 SNF 概率分布
  const probBox = document.getElementById("surv-snf-prob");
  if (r.snf_probabilities) {
    const parts = Object.entries(r.snf_probabilities)
      .map(([s, p]) => `${s}=${(p*100).toFixed(0)}%`).join(" · ");
    probBox.innerHTML = `Tab ① 预测的 SNF 概率: <b>${parts}</b>` +
      (r.auto_predicted_snf ? ` &nbsp;|&nbsp; argmax = ${r.auto_predicted_snf}` : "");
  } else {
    probBox.textContent = "";
  }

  // 绑定视图切换
  const modeSel = document.getElementById("surv-view-mode");
  const varSel = document.getElementById("surv-view-variant");
  modeSel.onchange = () => redrawSurvPrediction(r);
  varSel.onchange = () => redrawSurvPrediction(r);

  redrawSurvPrediction(r);

  // 系数表(用默认变体 snf+treat)
  const coefBox = document.getElementById("surv-coef");
  const vmeta = r.variants_meta || {};
  const defaultVk = r.default_variant || "snf+treat";
  const firstOk = Object.entries(r.endpoints).find(([_, v]) => v.variants[defaultVk] && !v.variants[defaultVk].error);
  if (firstOk) {
    const [ep, v] = firstOk;
    const vv = v.variants[defaultVk];
    const lab = (vmeta[defaultVk]||{}).label || defaultVk;
    coefBox.innerHTML = `<b>${ep} × ${escapeHTML(lab)}</b> Top 8 特征 (HR > 1 = 风险增加):<br>`
      + vv.top_coefficients.map(c =>
          `<code>${c.feature}</code>: HR = ${c.hazard_ratio.toFixed(2)}, p = ${c.p.toExponential(2)}`
        ).join(" &nbsp;|&nbsp; ");
  }
}

function redrawSurvPrediction(r) {
  const mode = document.getElementById("surv-view-mode").value;
  const picker = document.getElementById("surv-variant-picker");
  if (mode === "byvariant") {
    // 旧版:按变体对比
    const vorder = r.variants_order || ["base","treat","snf","snf+treat"];
    const vmeta = r.variants_meta || {};
    const colors = { "base":"#94a3b8", "treat":"#059669", "snf":"#2563eb", "snf+treat":"#dc2626" };
    picker.innerHTML = "";
    vorder.forEach(vk => {
      const lab = (vmeta[vk]||{}).label || vk;
      const l = document.createElement("label");
      l.innerHTML = `<input type="checkbox" data-variant="${vk}" ${SURV_SELECTED_VARIANTS.has(vk)?'checked':''}>
        <span style="display:inline-block;width:10px;height:10px;background:${colors[vk]};border-radius:2px;margin-right:4px;vertical-align:middle"></span>
        ${escapeHTML(lab)}`;
      l.style.marginRight = "16px";
      picker.appendChild(l);
    });
    picker.onchange = (e) => {
      if (e.target && e.target.dataset.variant) {
        const k = e.target.dataset.variant;
        if (e.target.checked) SURV_SELECTED_VARIANTS.add(k); else SURV_SELECTED_VARIANTS.delete(k);
        drawSurvByVariant(r, SURV_SELECTED_VARIANTS);
        renderSurvMilestonesByVariant(r, SURV_SELECTED_VARIANTS);
      }
    };
    drawSurvByVariant(r, SURV_SELECTED_VARIANTS);
    renderSurvMilestonesByVariant(r, SURV_SELECTED_VARIANTS);
  } else {
    // 新版:按 SNF 分型对比(默认)
    const vk = document.getElementById("surv-view-variant").value;
    const usesSnf = (vk === "snf" || vk === "snf+treat");
    picker.innerHTML = `<span class="hint">
      把你的病人依次假设成 SNF1/SNF2/SNF3/SNF4 分别跑当前 Cox 模型 (${escapeHTML(vk)}), 4 条曲线直接比较"如果我是这个亚型预后会怎样"。
      虚线 "Expected" = 按 Tab ① 预测的 SNF 概率加权平均。
      ${usesSnf ? "" : "<br><b>注意</b>:当前模型 <code>"+vk+"</code> 没把 SNF 当特征,4 条曲线必然重合 —— 可作为'加 SNF 之前没分叉'的对照。"}
      ${document.getElementById("surv-show-baseline").checked ? "<br>灰色虚线 = Baseline(临床特征 only) 模型对你的预测,作为'什么都不加时的参考'。" : ""}
    </span>`;
    picker.onchange = null;
    drawSurvBySubtype(r);
    renderSurvMilestonesBySubtype(r);
  }
}

function renderSurvMilestonesByVariant(r, selected) {
  const vorder = r.variants_order || ["base","treat","snf","snf+treat"];
  const vmeta = r.variants_meta || {};
  const msBox = document.getElementById("surv-milestones");
  let html = `<h3>关键时间点生存概率 (按变体)</h3><table class="data"><thead><tr>
    <th>端点</th><th>变体</th><th>Partial HR</th>
    <th>2 年</th><th>5 年</th><th>10 年</th>
    <th>Test C-index</th>
  </tr></thead><tbody>`;
  Object.entries(r.endpoints).forEach(([ep, epData]) => {
    vorder.forEach(vk => {
      if (!selected.has(vk)) return;
      const v = epData.variants[vk];
      if (!v || v.error) return;
      const p = v.prediction;
      const hr = p.partial_hazard;
      const hrTag = hr < 0.85 ? "color:#166534" : (hr > 1.15 ? "color:#b91c1c" : "");
      html += `<tr>
        <td><b>${ep}</b></td>
        <td>${escapeHTML((vmeta[vk]||{}).label || vk)}</td>
        <td style="${hrTag}">${hr.toFixed(2)}</td>
        <td>${(p.milestones.p_survive_24mo*100).toFixed(1)}%</td>
        <td>${(p.milestones.p_survive_60mo*100).toFixed(1)}%</td>
        <td>${(p.milestones.p_survive_120mo*100).toFixed(1)}%</td>
        <td>${v.performance.test_c_index.toFixed(3)}</td>
      </tr>`;
    });
  });
  html += "</tbody></table>";
  msBox.innerHTML = html;
}

function renderSurvMilestonesBySubtype(r) {
  const vk = document.getElementById("surv-view-variant").value;
  const msBox = document.getElementById("surv-milestones");
  const probs = r.snf_probabilities || {};
  let html = `<h3>关键时间点生存概率 (把病人假设为不同 SNF 亚型 | Cox 模型: ${vk})</h3>
    <table class="data"><thead><tr>
      <th>端点</th><th>亚型假设</th><th>Tab ① 概率</th><th>Partial HR</th>
      <th>2 年</th><th>5 年</th><th>10 年</th><th>Test C-index</th>
    </tr></thead><tbody>`;
  Object.entries(r.endpoints).forEach(([ep, epData]) => {
    const v = epData.variants[vk];
    if (!v || v.error || !v.by_subtype || v.by_subtype.error) {
      html += `<tr><td>${ep}</td><td colspan="7" style="color:#b91c1c">该变体不支持 SNF 切换</td></tr>`;
      return;
    }
    const labels = v.by_subtype.subtype_labels;
    labels.forEach(s => {
      const pred = v.by_subtype.per_subtype[s];
      const hr = pred.partial_hazard;
      const hrTag = hr < 0.85 ? "color:#166534" : (hr > 1.15 ? "color:#b91c1c" : "");
      const prob = probs[s];
      html += `<tr>
        <td><b>${ep}</b></td>
        <td><b>${s}</b></td>
        <td>${prob != null ? (prob*100).toFixed(0)+'%' : '-'}</td>
        <td style="${hrTag}">${hr.toFixed(2)}</td>
        <td>${(pred.milestones.p_survive_24mo*100).toFixed(1)}%</td>
        <td>${(pred.milestones.p_survive_60mo*100).toFixed(1)}%</td>
        <td>${(pred.milestones.p_survive_120mo*100).toFixed(1)}%</td>
        <td>${v.performance.test_c_index.toFixed(3)}</td>
      </tr>`;
    });
    if (v.by_subtype.expected) {
      const exp = v.by_subtype.expected;
      html += `<tr style="background:#f8fafc;font-style:italic">
        <td><b>${ep}</b></td>
        <td><b>Expected</b><br><small>(按概率加权)</small></td>
        <td>—</td><td>—</td>
        <td>${(exp.milestones.p_survive_24mo*100).toFixed(1)}%</td>
        <td>${(exp.milestones.p_survive_60mo*100).toFixed(1)}%</td>
        <td>${(exp.milestones.p_survive_120mo*100).toFixed(1)}%</td>
        <td>—</td>
      </tr>`;
    }
  });
  html += "</tbody></table>";
  msBox.innerHTML = html;
}

function drawSurvBySubtype(r) {
  const canvas = document.getElementById("surv-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const endpoints = Object.keys(r.endpoints);
  const vk = document.getElementById("surv-view-variant").value;
  const subtypeColors = { SNF1:"#2563eb", SNF2:"#059669", SNF3:"#d97706", SNF4:"#dc2626" };
  const maxT = 150;

  const nEp = endpoints.length, gap = 24;
  const subW = (W - gap * (nEp + 1)) / nEp;
  const subH = H - 76;

  endpoints.forEach((ep, ei) => {
    const ox = gap + ei * (subW + gap), oy = 30;
    ctx.fillStyle = "#111"; ctx.font = "12px sans-serif";
    ctx.fillText(`${ep} · ${r.endpoints[ep].label || ""}`, ox, oy - 10);

    // 网格 + 坐标
    for (let k = 0; k <= 10; k++) {
      const y = oy + subH - k/10 * subH;
      ctx.strokeStyle = "#eef2f7";
      ctx.beginPath(); ctx.moveTo(ox, y); ctx.lineTo(ox + subW, y); ctx.stroke();
    }
    ctx.strokeStyle = "#111"; ctx.beginPath();
    ctx.moveTo(ox, oy); ctx.lineTo(ox, oy + subH);
    ctx.lineTo(ox + subW, oy + subH); ctx.stroke();
    ctx.fillStyle = "#6b7280";
    for (let k = 0; k <= 5; k++) {
      const x = ox + subW * k / 5;
      ctx.fillText(`${Math.round(k*maxT/5)}`, x - 6, oy + subH + 12);
      const y = oy + subH - k/5 * subH;
      ctx.fillText((k/5).toFixed(1), ox - 28, y + 4);
    }
    ctx.fillText("mo", ox + subW - 18, oy + subH + 26);

    const v = r.endpoints[ep].variants[vk];
    if (!v || v.error || !v.by_subtype || v.by_subtype.error) {
      ctx.fillStyle = "#b91c1c";
      ctx.fillText("此变体不支持 SNF 切换", ox + 8, oy + 20);
      return;
    }
    // 可选: baseline 模型(base 变体)曲线作为灰色虚线参考
    if (document.getElementById("surv-show-baseline").checked && vk !== "base") {
      const baseV = r.endpoints[ep].variants["base"];
      if (baseV && !baseV.error && baseV.prediction) {
        ctx.strokeStyle = "#9ca3af";
        ctx.lineWidth = 1.6;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        baseV.prediction.times.forEach((t, i) => {
          const x = ox + Math.min(t, maxT) / maxT * subW;
          const y = oy + subH - baseV.prediction.survival[i] * subH;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    const bys = v.by_subtype;
    bys.subtype_labels.forEach(s => {
      const pred = bys.per_subtype[s];
      ctx.strokeStyle = subtypeColors[s] || "#333";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      pred.times.forEach((t, i) => {
        const x = ox + Math.min(t, maxT) / maxT * subW;
        const y = oy + subH - pred.survival[i] * subH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
    // expected 虚线
    if (bys.expected) {
      ctx.strokeStyle = "#1f2937"; ctx.lineWidth = 2.2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      bys.expected.times.forEach((t, i) => {
        const x = ox + Math.min(t, maxT) / maxT * subW;
        const y = oy + subH - bys.expected.survival[i] * subH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }
  });

  // 底部图例
  let lx = gap; const ly = H - 16;
  ctx.font = "11px sans-serif";
  ["SNF1","SNF2","SNF3","SNF4"].forEach(s => {
    ctx.fillStyle = subtypeColors[s];
    ctx.fillRect(lx, ly - 8, 14, 10);
    ctx.fillStyle = "#111";
    const prob = (r.snf_probabilities || {})[s];
    const tag = prob != null ? `${s} (p=${(prob*100).toFixed(0)}%)` : s;
    ctx.fillText(tag, lx + 18, ly);
    lx += ctx.measureText(tag).width + 32;
  });
  // Expected 图例
  ctx.strokeStyle = "#1f2937"; ctx.lineWidth = 2.2;
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(lx, ly - 3); ctx.lineTo(lx + 18, ly - 3); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#111";
  ctx.fillText("Expected (按概率加权)", lx + 22, ly);
  lx += ctx.measureText("Expected (按概率加权)").width + 32;
  // Baseline 图例
  if (document.getElementById("surv-show-baseline").checked) {
    ctx.strokeStyle = "#9ca3af"; ctx.lineWidth = 1.6;
    ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(lx, ly - 3); ctx.lineTo(lx + 18, ly - 3); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#111";
    ctx.fillText("Baseline (临床 only)", lx + 22, ly);
  }
}

function drawSurvByVariant(r, selected) {
  const canvas = document.getElementById("surv-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const endpoints = Object.keys(r.endpoints);
  const vorder = r.variants_order || ["base","treat","snf","snf+treat"];
  const vmeta = r.variants_meta || {};

  // 3 个端点一个一张小图,横向排布
  const nEp = endpoints.length;
  const gap = 20;
  const subW = (W - gap * (nEp + 1)) / nEp;
  const subH = H - 70;
  const variantColors = { "base":"#94a3b8", "treat":"#059669", "snf":"#2563eb", "snf+treat":"#dc2626" };
  const maxT = 150;

  endpoints.forEach((ep, ei) => {
    const ox = gap + ei * (subW + gap);
    const oy = 30;
    ctx.strokeStyle = "#111"; ctx.fillStyle = "#111"; ctx.font = "12px sans-serif";
    ctx.fillText(`${ep}  ·  ${r.endpoints[ep].label || ""}`, ox, oy - 10);
    // 网格
    for (let k = 0; k <= 10; k++) {
      const y = oy + subH - k/10 * subH;
      ctx.strokeStyle = "#eef2f7";
      ctx.beginPath(); ctx.moveTo(ox, y); ctx.lineTo(ox + subW, y); ctx.stroke();
    }
    ctx.strokeStyle = "#111";
    ctx.beginPath();
    ctx.moveTo(ox, oy); ctx.lineTo(ox, oy + subH);
    ctx.lineTo(ox + subW, oy + subH); ctx.stroke();
    ctx.fillStyle = "#6b7280";
    for (let k = 0; k <= 5; k++) {
      const x = ox + subW * k / 5;
      ctx.fillText(`${Math.round(k*maxT/5)}`, x - 6, oy + subH + 12);
    }
    ctx.fillText("mo", ox + subW - 18, oy + subH + 26);
    for (let k = 0; k <= 5; k++) {
      const y = oy + subH - k/5 * subH;
      ctx.fillText((k/5).toFixed(1), ox - 28, y + 4);
    }

    // 曲线
    const epData = r.endpoints[ep];
    vorder.forEach(vk => {
      if (!selected.has(vk)) return;
      const v = epData.variants[vk];
      if (!v || v.error) return;
      const p = v.prediction;
      ctx.strokeStyle = variantColors[vk] || "#333";
      ctx.lineWidth = 2;
      ctx.beginPath();
      p.times.forEach((t, i) => {
        const x = ox + Math.min(t, maxT) / maxT * subW;
        const y = oy + subH - p.survival[i] * subH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  });

  // 底部小图例
  let lx = gap; const ly = H - 14;
  ctx.font = "11px sans-serif";
  vorder.forEach(vk => {
    if (!selected.has(vk)) return;
    const c = variantColors[vk] || "#333";
    const lab = (vmeta[vk]||{}).label || vk;
    ctx.fillStyle = c;
    ctx.fillRect(lx, ly - 8, 12, 10);
    ctx.fillStyle = "#111";
    ctx.fillText(lab, lx + 16, ly);
    lx += ctx.measureText(lab).width + 34;
  });
}

init();
