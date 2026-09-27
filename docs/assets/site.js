// Static showcase: reads the pre-generated designs and the Phase 3 CSV.
// No backend. Everything comes from files under docs/.

(function () {
  "use strict";

  // Hide header links that haven't been set yet, rather than shipping dead ones.
  for (const id of ["link-github", "link-video"]) {
    const a = document.getElementById(id);
    if (a && !a.getAttribute("href")) a.hidden = true;
  }

  const fmt = (v, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const signed = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v), d) + "%";

  function el(tag, attrs, text) {
    const e = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  }

  // --- line chart: one series, crosshair + tooltip ---------------------------

  const SVGNS = "http://www.w3.org/2000/svg";
  function s(tag, attrs) {
    const e = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    return e;
  }

  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1;
    const step0 = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((m) => span / m <= n) || 10 * mag;
    const t = [];
    const top = Math.ceil(hi / step - 1e-9) * step;
    for (let v = Math.floor(lo / step) * step; v <= top + step * 1e-6; v += step) t.push(+v.toFixed(10));
    return t;
  }

  function lineChart(fig, rows, key, label, unit) {
    const W = 260, H = 150, m = { l: 34, r: 8, t: 8, b: 24 };
    const xs = rows.map((r) => r.r_R), ys = rows.map((r) => r[key]);
    let lo = Math.min(...ys), hi = Math.max(...ys);
    const pad = (hi - lo) * 0.1 || 1;
    const yt = niceTicks(key === "chord_in" ? 0 : lo - pad, hi + pad, 4);
    lo = yt[0]; hi = yt[yt.length - 1];
    const X = (v) => m.l + ((v - 0) / (1 - 0)) * (W - m.l - m.r);
    const Y = (v) => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);

    fig.appendChild(el("figcaption", null, `${label} (${unit}) vs radius r/R`));
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", tabindex: "0",
      "aria-label": `${label} along the blade, from ${fmt(ys[0], 2)} ${unit} at the root to ${fmt(ys[ys.length - 1], 2)} ${unit} at the tip` });

    const g = s("g", { class: "grid" });
    const ax = s("g", { class: "axis" });
    for (const v of yt) {
      g.appendChild(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
      const t = s("text", { x: m.l - 5, y: Y(v) + 3, "text-anchor": "end" });
      t.textContent = fmt(v, Math.abs(hi - lo) < 3 ? 1 : 0);
      ax.appendChild(t);
    }
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      const t = s("text", { x: X(v), y: H - 8, "text-anchor": "middle" });
      t.textContent = v.toString();
      ax.appendChild(t);
    }
    svg.append(g, ax);
    svg.appendChild(s("path", { class: "line", d: rows.map((r, i) => `${i ? "L" : "M"}${X(r.r_R).toFixed(1)},${Y(r[key]).toFixed(1)}`).join("") }));

    const hair = s("line", { class: "hair", y1: m.t, y2: H - m.b, visibility: "hidden" });
    const dot = s("circle", { class: "dot", r: 4, visibility: "hidden" });
    svg.append(hair, dot);
    fig.appendChild(svg);
    const tip = el("div", { class: "tip", hidden: "" });
    fig.appendChild(tip);

    let idx = -1;
    function show(i) {
      idx = Math.max(0, Math.min(rows.length - 1, i));
      const r = rows[idx], px = X(r.r_R), py = Y(r[key]);
      hair.setAttribute("x1", px); hair.setAttribute("x2", px);
      dot.setAttribute("cx", px); dot.setAttribute("cy", py);
      hair.setAttribute("visibility", "visible"); dot.setAttribute("visibility", "visible");
      tip.replaceChildren(el("b", null, `${fmt(r[key], key === "chord_in" ? 2 : 1)}${unit === "\u00b0" ? "" : " "}${unit}`), document.createTextNode(`  at r/R ${fmt(r.r_R, 2)}`));
      const box = svg.getBoundingClientRect(), scale = box.width / W;
      tip.style.left = Math.min(Math.max(px * scale, 50), box.width - 50) + "px";
      tip.style.top = (box.top - fig.getBoundingClientRect().top + py * scale) + "px";
      tip.hidden = false;
    }
    function hide() {
      hair.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden");
      tip.hidden = true;
    }
    svg.addEventListener("pointermove", (e) => {
      const box = svg.getBoundingClientRect();
      const x = ((e.clientX - box.left) / box.width) * W;
      let best = 0;
      xs.forEach((v, i) => { if (Math.abs(X(v) - x) < Math.abs(X(xs[best]) - x)) best = i; });
      show(best);
    });
    svg.addEventListener("pointerleave", hide);
    svg.addEventListener("blur", hide);
    svg.addEventListener("focus", () => show(idx < 0 ? 0 : idx));
    svg.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") { show(idx + 1); e.preventDefault(); }
      if (e.key === "ArrowLeft") { show(idx - 1); e.preventDefault(); }
    });
  }

  // --- design cards ----------------------------------------------------------

  function card(d) {
    const base = `designs/${d.name}/`;
    const node = document.getElementById("card-tpl").content.cloneNode(true);
    const q = (sel) => node.querySelector(sel);
    const inp = d.inputs;

    const mv = q("model-viewer");
    mv.setAttribute("src", base + d.files.glb);
    mv.setAttribute("alt", `3D model of the ${inp.diameter_in} inch, ${d.blade_count}-blade design`);

    q(".card-title").textContent = `${inp.diameter_in} in, ${d.blade_count} blades`;
    q(".card-req").textContent = `Target ${fmt(inp.thrust_target_gf)} gf at ${fmt(inp.rpm)} rpm`;
    q(".card-blurb").textContent = d.blurb;

    for (const w of d.warnings) q(".warnings").appendChild(el("p", { class: "warning", role: "note" }, w));

    const metrics = [
      ["Thrust", `${fmt(d.thrust_gf)} gf`, signed(d.thrust_error_pct)],
      ["Shaft power", `${fmt(d.shaft_power_W, 1)} W`, null],
      ["Efficiency", `${fmt(d.gf_per_W, 2)} gf/W`, null],
      ["PLA mass", `${fmt(d.mass_pla_g, 1)} g`, "est."],
    ];
    for (const [k, v, sub] of metrics) {
      const wrap = el("div");
      wrap.appendChild(el("dt", null, k));
      const dd = el("dd", null, v + " ");
      if (sub) dd.appendChild(el("small", null, sub));
      wrap.appendChild(dd);
      q(".metrics").appendChild(wrap);
    }

    q(".blade-note").textContent = d.blade_count_forced
      ? (d.auto_blade_count === d.blade_count
          ? `Blade count set to ${d.blade_count}. Auto-select picks the same.`
          : `Blade count forced to ${d.blade_count}. Auto-select would pick ${d.auto_blade_count}.`)
      : `Blade count auto-selected from 2 to 6.`;

    q(".charts").querySelectorAll(".chart").forEach((fig) =>
      lineChart(fig, d.geometry, fig.dataset.key, fig.dataset.label, fig.dataset.unit));

    const tb = q(".cands tbody");
    for (const c of d.candidates) {
      const tr = el("tr", c.blades === d.blade_count ? { class: "chosen" } : null);
      tr.append(
        el("td", null, c.blades + (c.blades === d.blade_count ? " (shown)" : "")),
        el("td", { class: "num" }, fmt(c.thrust_gf)),
        el("td", { class: "num" }, signed(c.error_pct)),
        el("td", { class: "num" }, fmt(c.gf_per_W, 2)),
        el("td", null, c.meets ? "yes" : "no"));
      tb.appendChild(tr);
    }

    const dl = q(".downloads");
    for (const [k, label] of [["stl", "STL (print)"], ["step", "STEP (CAD)"], ["png", "Render"]]) {
      dl.appendChild(el("a", { class: "btn", href: base + d.files[k], download: "" }, label));
    }
    dl.appendChild(el("a", { class: "btn", href: base + "design.json" }, "design.json"));
    return node;
  }

  async function loadDesigns() {
    const host = document.getElementById("cards");
    try {
      const names = await (await fetch("designs/index.json")).json();
      const designs = await Promise.all(names.map((n) => fetch(`designs/${n}/design.json`).then((r) => r.json())));
      host.replaceChildren(...designs.map(card));
    } catch (err) {
      host.replaceChildren(el("p", { class: "muted" }, "Couldn't load the example designs. If you opened this file directly, serve the docs/ folder over HTTP instead."));
      console.error(err);
    }
  }

  // --- Phase 3 table ---------------------------------------------------------

  async function loadPhase3() {
    const tbody = document.querySelector("#phase3 tbody");
    try {
      const text = await (await fetch("data/phase3_validation.csv")).text();
      const [head, ...lines] = text.trim().split(/\r?\n/);
      const cols = head.split(",");
      const rows = lines.map((l) => Object.fromEntries(l.split(",").map((v, i) => [cols[i], v])));
      tbody.replaceChildren();
      for (const r of rows) {
        const D = +r.D_in, vs = +r.vs_ref_pct;
        const inRange = D >= 6 && D <= 11;
        const tr = el("tr", inRange ? null : { class: "out" });
        const bl = `${r.B_chosen} / ${r.B_ref}`;
        const verdict = el("span", { class: "pill " + (!inRange ? "na" : vs >= -15 ? "pass" : "fail") },
          !inRange ? "outside 6–11 in" : vs >= -15 ? "pass" : "fail");
        const vtd = el("td"); vtd.appendChild(verdict);
        tr.append(
          el("td", null, r.case),
          el("td", { class: "num" }, fmt(D, D % 1 ? 1 : 0)),
          el("td", { class: "num" }, fmt(+r.target_gf)),
          el("td", { class: "num" }, signed(+r.thrust_err_pct)),
          el("td", { class: "num" }, bl),
          el("td", { class: "num" }, fmt(+r.design_gfW, 1)),
          el("td", { class: "num" }, fmt(+r.ref_gfW, 1)),
          el("td", { class: "num" }, signed(vs, 0)),
          vtd);
        tbody.appendChild(tr);
      }
    } catch (err) {
      tbody.replaceChildren();
      const tr = el("tr"); tr.appendChild(el("td", { colspan: "9", class: "muted" }, "Couldn't load the Phase 3 results."));
      tbody.appendChild(tr);
      console.error(err);
    }
  }

  loadDesigns();
  loadPhase3();
})();
