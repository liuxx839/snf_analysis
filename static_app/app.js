/**
 * 纯前端推理:
 *   1) 拉 models.json (导出的系数 + 预处理参数 + baseline survival)
 *   2) 表单 → 标准化数值 + One-Hot 类别 → 与 LR 系数点积 → softmax 给 SNF 概率
 *   3) Cox: S(t) = S0(t) ^ exp(beta · x), 4 个变体 + 4 种 SNF 假设
 *
 * 没有任何后端调用。模型系数都已经在浏览器里。
 */

const SNF_COLORS = { SNF1: "#2563eb", SNF2: "#10b981", SNF3: "#f59e0b", SNF4: "#ef4444" };

let M = null;        // 全部模型 + 元数据
let SURV_CHART = null;

// ---------- 读模型 ----------
fetch("models.json")
  .then(r => r.json())
  .then(data => { M = data; init(); })
  .catch(e => alert("无法加载 models.json: " + e.message));

// ---------- 通用工具 ----------
function softmax(arr) {
  const max = Math.max(...arr);
  const e = arr.map(v => Math.exp(v - max));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}
function lerp(t, ts, ys) {
  if (t <= ts[0]) return ys[0];
  for (let i = 1; i < ts.length; i++) {
    if (t <= ts[i]) {
      const r = (t - ts[i-1]) / (ts[i] - ts[i-1] || 1);
      return ys[i-1] + r * (ys[i] - ys[i-1]);
    }
  }
  return ys[ys.length - 1];
}

// ---------- 预处理(数值+OHE)生成长向量 ----------
function buildFeatureVector(patient, modelMeta) {
  const numeric = modelMeta.numeric_features || modelMeta.numeric;
  const categorical = modelMeta.categorical_features || modelMeta.categorical;
  const numMeta = modelMeta.numeric_meta;
  const catMeta = modelMeta.categorical_meta;
  const namesOut = modelMeta.feature_names_out;
  const usedCols = modelMeta.used_columns; // 仅 cox 用

  // 1) 先按训练时拼出全部特征
  // 输出顺序 = num__<col1>, num__<col2>, ..., cat__<colA>_<val1>, cat__<colA>_<val2>, ...
  const vec = {};
  numeric.forEach(c => {
    let v = patient[c];
    if (v === "" || v === null || v === undefined || isNaN(parseFloat(v))) v = numMeta[c].median;
    else v = parseFloat(v);
    const z = (v - numMeta[c].mean) / (numMeta[c].std || 1);
    vec[`num__${c}`] = z;
  });
  categorical.forEach(c => {
    let v = patient[c];
    if (v === "" || v === null || v === undefined) v = "Missing";
    else v = String(v);
    catMeta[c].forEach(cat => {
      vec[`cat__${c}_${cat}`] = (cat === v) ? 1 : 0;
    });
  });

  if (namesOut) {
    return namesOut.map(n => vec[n] ?? 0);
  } else if (usedCols) {
    return usedCols.map(n => vec[n] ?? 0);
  }
  return Object.values(vec);
}

// ---------- 分类预测 ----------
function predictSNF(patient) {
  const m = M.classifier;
  const x = buildFeatureVector(patient, m);
  const logits = m.coef_.map((row, k) => {
    let s = m.intercept_[k];
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    return s;
  });
  const probs = softmax(logits);
  const out = {};
  m.classes_.forEach((c, k) => out[c] = probs[k]);
  return out;
}

// ---------- 生存预测 ----------
function predictSurvival(patient, endpoint, variantKey, cohort) {
  const variant = M.survival.cohorts[cohort][endpoint][variantKey];
  if (!variant || variant.error) return null;
  const x = buildFeatureVector(patient, variant);
  let lp = 0;
  for (let i = 0; i < x.length; i++) lp += variant.coef_[i] * x[i];
  const hr = Math.exp(lp);
  const ts = variant.baseline_times;
  const s0 = variant.baseline_survival;
  const surv = s0.map(v => Math.pow(v, hr));
  return { ts, surv, hr };
}

// ---------- 4 SNF 假设 ----------
function predictSurvivalBySubtype(patient, endpoint, variantKey, cohort, snfProbs) {
  const labels = ["SNF1", "SNF2", "SNF3", "SNF4"];
  const per = {};
  labels.forEach(s => {
    const pt = { ...patient, SNF_subtype: s };
    per[s] = predictSurvival(pt, endpoint, variantKey, cohort);
  });
  // expected = Σ w_i * S_i(t)
  if (snfProbs && per.SNF1) {
    const T = per.SNF1.surv.length;
    const exp = new Array(T).fill(0);
    labels.forEach(s => {
      const w = snfProbs[s] || 0;
      per[s].surv.forEach((v, i) => exp[i] += w * v);
    });
    per.expected = { ts: per.SNF1.ts, surv: exp };
  }
  return per;
}

// ============================ UI ============================

// 表单分组 + 友好 label
const FORM_GROUPS = [
  {
    title: "基本信息",
    fields: [
      { name: "Age", label: "年龄 (岁)", type: "number" },
      { name: "Menopause", label: "绝经状态", type: "select", options: ["No", "Yes"], optionLabels: ["未绝经", "已绝经"] },
    ]
  },
  {
    title: "肿瘤特征",
    fields: [
      { name: "Tumor_size_cm", label: "肿瘤最大径 (cm)", type: "number", step: "0.1" },
      { name: "Grade", label: "病理分级", type: "select", options: ["1", "2", "3"], optionLabels: ["Grade 1 (高分化)", "Grade 2 (中分化)", "Grade 3 (低分化)"] },
      { name: "Ki67", label: "Ki-67 (%)", type: "number", step: "1", hint: "0–100,反映增殖活性" },
      { name: "pT", label: "T 分期", type: "select", options: ["pT1", "pT2", "pT3"] },
    ]
  },
  {
    title: "淋巴结",
    fields: [
      { name: "Positive_axillary_lymph_nodes", label: "阳性腋窝淋巴结数", type: "number" },
      { name: "pN", label: "N 分期", type: "select", options: ["pN0", "pN1", "pN2", "pN3"] },
    ]
  },
  {
    title: "免疫组化",
    fields: [
      { name: "ER_percent", label: "ER 阳性比例 (%)", type: "number", step: "1" },
      { name: "PR_status", label: "PR 状态", type: "select", options: ["Positive", "Negative"], optionLabels: ["阳性", "阴性"] },
      { name: "PR_percent", label: "PR 阳性比例 (%)", type: "number", step: "1" },
      { name: "HER2_IHC_Status", label: "HER2 IHC 评分", type: "select", options: ["0", "1", "2"], optionLabels: ["0", "1+", "2+ (FISH-)"] },
    ]
  },
  {
    title: "其它(可选)",
    fields: [
      { name: "PAM50", label: "PAM50 分型 (如做过)", type: "select", options: ["", "LumA", "LumB", "Her2", "Basal", "Normal"], optionLabels: ["未做", "LumA", "LumB", "Her2", "Basal", "Normal"] },
    ]
  },
];

const EXAMPLE = {
  Age: 45, Tumor_size_cm: 2.2, Positive_axillary_lymph_nodes: 0,
  ER_percent: 90, PR_percent: 80, Ki67: 20, HER2_IHC_Status: "1",
  Menopause: "No", Grade: "2", pT: "pT2", pN: "pN0", PR_status: "Positive",
  PAM50: ""
};

function init() {
  buildForm();
  renderSNFDescriptions();
  renderBenchmarks();

  document.getElementById("btn-predict").addEventListener("click", onPredict);
  document.getElementById("btn-example").addEventListener("click", () => fillForm(EXAMPLE));
  document.getElementById("btn-clear").addEventListener("click", clearForm);
  document.getElementById("surv-endpoint").addEventListener("change", onPredict);
  document.getElementById("surv-with-treat").addEventListener("change", onPredict);
}

function buildForm() {
  const f = document.getElementById("patient-form");
  f.innerHTML = "";
  FORM_GROUPS.forEach(g => {
    const h = document.createElement("div");
    h.className = "field section-title";
    h.textContent = g.title;
    f.appendChild(h);
    g.fields.forEach(fld => {
      const w = document.createElement("div");
      w.className = "field";
      const lab = document.createElement("label");
      lab.textContent = fld.label;
      lab.htmlFor = "in-" + fld.name;
      w.appendChild(lab);
      let el;
      if (fld.type === "select") {
        el = document.createElement("select");
        fld.options.forEach((o, i) => {
          const opt = document.createElement("option");
          opt.value = o;
          opt.textContent = (fld.optionLabels && fld.optionLabels[i]) || o || "(未填)";
          el.appendChild(opt);
        });
        // 默认 "未填"(第一个空选项), 或者保持 placeholder
        if (!fld.options.includes("")) {
          // 加一个空的
          const empty = document.createElement("option");
          empty.value = ""; empty.textContent = "(未填)";
          el.insertBefore(empty, el.firstChild);
          el.value = "";
        }
      } else {
        el = document.createElement("input");
        el.type = fld.type;
        if (fld.step) el.step = fld.step;
        el.placeholder = "(未填)";
      }
      el.id = "in-" + fld.name;
      el.name = fld.name;
      w.appendChild(el);
      if (fld.hint) {
        const h2 = document.createElement("small");
        h2.className = "hint";
        h2.textContent = fld.hint;
        w.appendChild(h2);
      }
      f.appendChild(w);
    });
  });
}

function fillForm(data) {
  Object.entries(data).forEach(([k, v]) => {
    const el = document.querySelector(`[name="${k}"]`);
    if (el) el.value = v ?? "";
  });
}
function clearForm() {
  document.querySelectorAll("#patient-form [name]").forEach(el => el.value = "");
}
function collectPatient() {
  const out = {};
  document.querySelectorAll("#patient-form [name]").forEach(el => {
    out[el.name] = el.value;
  });
  return out;
}

// ----- predict pipeline -----
function onPredict() {
  if (!M) return;
  const patient = collectPatient();
  const probs = predictSNF(patient);
  renderSNFResult(probs);
  renderSurvivalResult(patient, probs);
  renderModelPerf();
  document.getElementById("result-snf").classList.remove("hidden");
  document.getElementById("result-surv").classList.remove("hidden");
  document.getElementById("result-bench").classList.remove("hidden");
  // 滚到结果
  document.getElementById("result-snf").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSNFResult(probs) {
  const order = ["SNF1", "SNF2", "SNF3", "SNF4"];
  const labelText = {
    SNF1: "经典 Luminal", SNF2: "免疫激活型",
    SNF3: "高增殖型", SNF4: "RTK 驱动型",
  };
  const box = document.getElementById("snf-bars");
  box.innerHTML = "";
  order.forEach(s => {
    const p = probs[s] || 0;
    const row = document.createElement("div");
    row.className = "snf-bar";
    row.style.setProperty("--snf-color", SNF_COLORS[s]);
    row.innerHTML = `
      <div class="label">${s}<small>${labelText[s]}</small></div>
      <div class="track"><div class="fill" style="width:${(p*100).toFixed(1)}%"></div></div>
      <div class="pct">${(p*100).toFixed(1)}%</div>
    `;
    box.appendChild(row);
  });

  // 最可能亚型解读
  const best = order.reduce((a, b) => probs[a] > probs[b] ? a : b);
  const second = order.filter(s => s !== best).reduce((a, b) => probs[a] > probs[b] ? a : b);
  const margin = probs[best] - probs[second];
  let conf = "中等";
  if (probs[best] >= 0.55 && margin >= 0.2) conf = "较高";
  else if (probs[best] < 0.4 || margin < 0.1) conf = "偏低 (边界病例)";

  document.getElementById("snf-most-likely").innerHTML = `
    <p>最可能的亚型: <strong style="color:${SNF_COLORS[best]}">${best} · ${labelText[best]}</strong>
       &nbsp;<span class="hint-pill">置信度: ${conf}</span></p>
    <p style="margin:6px 0 0;color:#374151">${M.label_description[best] || ""}</p>
  `;
}

function renderSNFDescriptions() {
  const order = ["SNF1", "SNF2", "SNF3", "SNF4"];
  const box = document.getElementById("snf-descriptions");
  box.innerHTML = "";
  order.forEach(s => {
    const c = document.createElement("div");
    c.className = "snf-desc-card";
    c.style.setProperty("--snf-color", SNF_COLORS[s]);
    c.innerHTML = `<h4>${s}</h4><p>${M.label_description[s] || ""}</p>`;
    box.appendChild(c);
  });
}

function renderModelPerf() {
  const m = M.classifier.performance;
  const cnt = m.class_counts || {};
  const lines = [];
  lines.push(`<p>训练样本 N = <b>${m.n_samples}</b>(SNF1=${cnt.SNF1||0}, SNF2=${cnt.SNF2||0}, SNF3=${cnt.SNF3||0}, SNF4=${cnt.SNF4||0})</p>`);
  lines.push(`<p>5 折交叉验证表现: <b>Weighted AUC = ${m.weighted_auc.toFixed(2)}</b>
    (95% CI ${m.weighted_auc_ci[0].toFixed(2)}–${m.weighted_auc_ci[1].toFixed(2)})</p>`);
  const order = ["SNF1","SNF2","SNF3","SNF4"];
  lines.push(`<p>每类 AUC: ` + order.map(s =>
    `<b style="color:${SNF_COLORS[s]}">${s}=${m.per_class_auc[s].toFixed(2)}</b>`
  ).join(" · ") + `</p>`);
  lines.push(`<p style="color:#64748b;font-size:13px">
    AUC = 1.0 表示完美区分,0.5 = 抛硬币。我们的模型只用临床特征,大概在 0.6–0.8 区间;
    原文用转录组测序能到 ~0.9。</p>`);
  document.getElementById("model-perf").innerHTML = lines.join("");
}

function renderBenchmarks() {
  const m = M.classifier.performance;
  const order = ["SNF1","SNF2","SNF3","SNF4"];
  let html = `<table class="bench-table"><thead><tr>
    <th class="label">模型</th><th>SNF1</th><th>SNF2</th><th>SNF3</th><th>SNF4</th><th>平均</th>
  </tr></thead><tbody>`;
  // 本工具
  const myAvg = order.reduce((a, s) => a + m.per_class_auc[s], 0) / 4;
  html += `<tr class="me"><td class="label">本工具(只用临床)</td>`;
  order.forEach(s => html += `<td>${m.per_class_auc[s].toFixed(2)}</td>`);
  html += `<td>${myAvg.toFixed(2)}</td></tr>`;
  // 原文 RF
  const rf = M.paper_benchmarks["Transcriptomics RF"];
  const rfAvg = order.reduce((a, s) => a + rf[s], 0) / 4;
  html += `<tr class="rf"><td class="label">原文 · 转录组随机森林</td>`;
  order.forEach(s => html += `<td>${rf[s].toFixed(2)}</td>`);
  html += `<td>${rfAvg.toFixed(2)}</td></tr>`;
  // 原文 CNN
  const cnn = M.paper_benchmarks["Pathology CNN"];
  const cnnAvg = order.reduce((a, s) => a + cnn[s], 0) / 4;
  html += `<tr class="cnn"><td class="label">原文 · 数字病理 CNN</td>`;
  order.forEach(s => html += `<td>${cnn[s].toFixed(2)}</td>`);
  html += `<td>${cnnAvg.toFixed(2)}</td></tr>`;
  html += `</tbody></table>`;
  html += `<p style="color:#64748b;font-size:13px;margin-top:10px">
    我们追不平转录组(0.89)和病理 CNN(0.81)是预期的 —— 那些方法用的是基因表达谱或显微切片图像,
    信息量远超几个数字。但本工具的优势是<b>不需要任何检测</b>,只看常规病理报告。</p>`;
  document.getElementById("bench-table").innerHTML = html;
}

// ----- survival -----
function renderSurvivalResult(patient, snfProbs) {
  const ep = document.getElementById("surv-endpoint").value;
  const vk = document.getElementById("surv-with-treat").value;
  const cohort = "matched";  // 始终用 matched (n 一致, 公平)

  const usesSnf = (vk === "snf" || vk === "snf+treat");
  const per = predictSurvivalBySubtype(patient, ep, vk, cohort, snfProbs);
  if (!per.SNF1) {
    document.getElementById("surv-summary").innerHTML = `<p style="color:#b91c1c">该端点的此变体训练失败,请换一个。</p>`;
    return;
  }
  const variant = M.survival.cohorts[cohort][ep][vk];
  drawSurvivalChart(per, ep, usesSnf);

  // milestones
  const order = ["SNF1","SNF2","SNF3","SNF4"];
  const ts = per.SNF1.ts;
  const sb = document.getElementById("surv-summary");
  sb.innerHTML = "";
  order.forEach(s => {
    const surv = per[s].surv;
    const v10 = lerp(120, ts, surv);
    const v5 = lerp(60, ts, surv);
    const card = document.createElement("div");
    card.className = "milestone";
    card.style.setProperty("--snf-color", SNF_COLORS[s]);
    card.innerHTML = `
      <div class="top"><span>假设你是 <b>${s}</b></span><span class="dot"></span></div>
      <div class="v">${(v10 * 100).toFixed(1)}%</div>
      <div class="sub">10 年生存率 · 5 年: ${(v5*100).toFixed(1)}%</div>
    `;
    sb.appendChild(card);
  });
  if (per.expected) {
    const v10 = lerp(120, ts, per.expected.surv);
    const card = document.createElement("div");
    card.className = "milestone";
    card.style.setProperty("--snf-color", "#374151");
    card.innerHTML = `
      <div class="top"><span><b>Expected</b> (按概率加权)</span><span class="dot"></span></div>
      <div class="v">${(v10 * 100).toFixed(1)}%</div>
      <div class="sub">综合 4 种亚型可能性后的期望</div>
    `;
    sb.appendChild(card);
  }

  // explainer about why curves overlap when no SNF
  if (!usesSnf) {
    const note = document.createElement("div");
    note.style.cssText = "margin-top:12px;padding:10px 14px;border-radius:8px;background:#fef3c7;color:#92400e;font-size:13px";
    note.innerHTML = `当前 Cox 模型不使用 SNF 字段,所以 4 条曲线<b>完全重合</b> —— 这正是"如果不告诉模型 SNF,亚型就完全不影响个人预测"的对照。`;
    sb.appendChild(note);
  }

  // perf line
  const perfLine = document.createElement("div");
  perfLine.style.cssText = "grid-column:1/-1;text-align:center;color:#64748b;font-size:12px;margin-top:6px";
  perfLine.innerHTML = `模型表现 (matched cohort, n=${variant.performance.n_total},
    事件=${variant.performance.n_events}):
    Test C-index = <b>${variant.performance.test_c_index.toFixed(2)}</b> ·
    CV C-index = ${variant.performance.cv_c_index.toFixed(2)}
    [${variant.performance.cv_c_index_ci[0].toFixed(2)}, ${variant.performance.cv_c_index_ci[1].toFixed(2)}]`;
  sb.appendChild(perfLine);
}

function drawSurvivalChart(per, endpoint, usesSnf) {
  const order = ["SNF1","SNF2","SNF3","SNF4"];
  const ts = per.SNF1.ts;

  // 把时间限制在 0..150 个月看起来好
  const idxMax = ts.findIndex(t => t > 150);
  const xs = ts.slice(0, idxMax > 0 ? idxMax : ts.length);

  const datasets = order.map(s => ({
    label: s,
    data: xs.map((t, i) => ({ x: t, y: per[s].surv[i] })),
    borderColor: SNF_COLORS[s],
    backgroundColor: SNF_COLORS[s] + "22",
    borderWidth: 2.5,
    pointRadius: 0,
    tension: 0.3,
    fill: false,
  }));

  if (per.expected) {
    datasets.push({
      label: "Expected (加权)",
      data: xs.map((t, i) => ({ x: t, y: per.expected.surv[i] })),
      borderColor: "#374151",
      borderWidth: 2,
      borderDash: [6, 4],
      pointRadius: 0,
      tension: 0.3,
      fill: false,
    });
  }

  if (SURV_CHART) SURV_CHART.destroy();
  const ctx = document.getElementById("surv-chart").getContext("2d");
  SURV_CHART = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      animation: { duration: 350 },
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 14, usePointStyle: true, padding: 14, font: { size: 13 } },
        },
        tooltip: {
          callbacks: {
            title: items => `${items[0].parsed.x.toFixed(0)} 月`,
            label: c => `${c.dataset.label}: ${(c.parsed.y * 100).toFixed(1)}% 生存`,
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          min: 0, max: 150,
          title: { display: true, text: "时间(月)" },
          grid: { color: "#eef2f7" },
          ticks: { stepSize: 30 },
        },
        y: {
          min: 0, max: 1,
          title: { display: true, text: `${endpoint} 生存概率` },
          grid: { color: "#eef2f7" },
          ticks: { callback: v => (v * 100).toFixed(0) + "%" },
        },
      },
    },
  });
}
