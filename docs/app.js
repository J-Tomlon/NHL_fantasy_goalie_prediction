// Crease Check — static front end for predictions.json (GitHub Pages friendly)
(() => {
  const KEY = "creaseCheck.myGoalies";
  const $ = (s, el = document) => el.querySelector(s);
  let DATA = null, byId = new Map(), mine = [], sort = { k: "prob", asc: false };

  const store = {
    get() { try { const v = JSON.parse(localStorage.getItem(KEY)); return Array.isArray(v) ? v : null; } catch { return null; } },
    set(v) { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch { /* private mode */ } },
  };
  const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
  const tone = (p) => (p >= 0.6 ? "good" : p >= 0.45 ? "mid" : "bad");
  const toneColor = { good: "var(--good)", mid: "var(--mid)", bad: "var(--bad)" };
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function when(g) {
    if (g.start_utc) {
      const d = new Date(g.start_utc);
      if (!isNaN(d)) return d.toLocaleString(undefined, { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
    }
    const [y, m, d] = g.date.split("-").map(Number);
    return new Date(y, m - 1, d).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
  }

  // ---------- header + footer
  function renderScoring(s) {
    const names = { W: "Win", GA: "GA", SV: "Save", SO: "Shutout", OTL: "OTL" };
    $("#scoring").innerHTML = Object.entries(s).map(([k, v]) =>
      `<span class="chip">${names[k] || k} <b>${v > 0 ? "+" : ""}${v}</b></span>`).join("");
  }
  function renderMetrics(m, generated) {
    const el = $("#metrics");
    const upd = generated ? new Date(generated).toLocaleString() : "unknown";
    if (!m || !m.model) { el.innerHTML = `<p class="metric-line">Updated ${esc(upd)}.</p>`; return; }
    const s = String(m.test_season);
    const season = `${s.slice(0, 4)}–${s.slice(6)}`;
    el.innerHTML = `<p class="metric-line"><strong>Updated ${esc(upd)}.</strong>
      Back-tested on the ${season} season (${m.model.n.toLocaleString()} starts, trained only on earlier seasons):
      calls were right ${pct(m.model.accuracy)} of the time; Brier score ${m.model.brier.toFixed(3)}
      vs ${m.baseline_career_rate.brier.toFixed(3)} for "just use his career rate" (lower is better).
      ${pct(m.positive_rate)} of all starts that season were positive.</p>`;
  }

  // ---------- cards
  function card(p) {
    const el = $("#card-tpl").content.firstElementChild.cloneNode(true);
    el.dataset.id = p.id;
    $(".remove", el).onclick = () => toggle(p.id, false);
    $(".name", el).textContent = p.name;
    const img = $(".shot", el);
    if (p.headshot) img.src = p.headshot; else img.style.visibility = "hidden";

    if (p.error) {
      $(".matchup", el).innerHTML = `<b>${esc(p.team || "")}</b> · ${esc(p.error)}`;
      $(".gauge", el).remove(); $(".verdict-row", el).remove(); $(".signals", el).remove();
      $(".stats", el).remove(); $(".h2h", el).remove();
      return el;
    }
    const g = p.next_game;
    $(".matchup", el).innerHTML = `<b>${esc(p.team)}</b> ${g.home ? "vs" : "@"} <b>${esc(g.opp)}</b> · ${esc(when(g))}`;
    const t = tone(p.prob_positive);
    const arc = $(".arc", el);
    arc.style.stroke = toneColor[t];
    arc.style.strokeDashoffset = 314.16 * (1 - p.prob_positive);
    $(".pct", el).textContent = pct(p.prob_positive);
    $(".gauge", el).setAttribute("aria-label", `${pct(p.prob_positive)} chance of a positive game`);
    const v = $(".verdict", el); v.textContent = p.verdict; v.classList.add(`t-${t}`);
    $(".exp", el).innerHTML = `Expected <b>${p.expected_points >= 0 ? "+" : ""}${p.expected_points.toFixed(1)}</b> pts`;

    if (p.starter_share_last10 != null && p.starter_share_last10 < 0.6) {
      const w = $(".warn", el);
      w.hidden = false;
      w.textContent = `⚠ Started ${pct(p.starter_share_last10)} of his team's last 10 — confirm he's in net. Prediction assumes he starts.`;
    }
    $(".signals", el).innerHTML = (p.signals || []).map((s) =>
      `<li class="${s.tone}"><span class="dot">${s.tone === "good" ? "+" : s.tone === "bad" ? "−" : "·"}</span>
       <span><strong>${esc(s.label)}</strong> <span class="d">${esc(s.detail)}</span></span></li>`).join("");

    const rows = [["Career", p.career], ["This season", p.season], [`vs ${g.opp}`, p.vs_opponent]];
    $(".stats tbody", el).innerHTML = rows.map(([label, s]) => !s || !s.starts
      ? `<tr><td>${esc(label)}</td><td>0</td><td colspan="4" style="text-align:left;color:var(--muted)">no starts</td></tr>`
      : `<tr><td>${esc(label)}</td><td>${s.starts}</td><td>${s.record}</td><td>${s.save_pct != null ? s.save_pct.toFixed(3).replace(/^0/, "") : "–"}</td>
         <td>${pct(s.pos_rate)}</td><td>${s.avg_points != null ? s.avg_points.toFixed(2) : "–"}</td></tr>`).join("");

    const games = p.vs_opponent.games || [];
    const det = $(".h2h", el);
    if (!games.length) det.remove();
    else {
      $("summary", det).textContent = `Last ${games.length} starts vs ${g.opp}`;
      $(".games", det).innerHTML = games.map((r) =>
        `<li><span>${r.date}</span><span>${esc(r.decision)}</span><span>${r.saves} SV · ${r.ga} GA${r.so ? " · SO" : ""}</span>
         <span class="${r.points > 0 ? "pos" : "neg"}">${r.points > 0 ? "+" : ""}${r.points.toFixed(1)}</span></li>`).join("");
    }
    return el;
  }

  function renderCards() {
    const wrap = $("#cards");
    wrap.replaceChildren(...mine.map((id) => byId.get(id)).filter(Boolean).map(card));
    $("#empty").hidden = mine.length > 0;
  }

  // ---------- big table
  function rowVal(p, k) {
    if (k === "name") return p.name;
    if (k === "team") return p.team || "";
    if (k === "opp") return p.next_game ? p.next_game.date + p.next_game.opp : "~";
    if (k === "share") return p.starter_share_last10 ?? -1;
    if (k === "prob") return p.prob_positive ?? -1;
    if (k === "exp") return p.expected_points ?? -99;
  }
  function renderTable() {
    const rows = DATA.goalies.slice().sort((a, b) => {
      const x = rowVal(a, sort.k), y = rowVal(b, sort.k);
      const c = typeof x === "string" ? x.localeCompare(y) : x - y;
      return sort.asc ? c : -c;
    });
    document.querySelectorAll("#all th").forEach((th) => {
      th.classList.toggle("sorted", th.dataset.k === sort.k);
      th.classList.toggle("asc", th.dataset.k === sort.k && sort.asc);
    });
    $("#all tbody").innerHTML = rows.map((p) => {
      const g = p.next_game;
      const next = g ? `${g.home ? "vs" : "@"} ${esc(g.opp)} <span class="muted">${g.date.slice(5).replace("-", "/")}</span>` : `<span class="muted">${esc(p.error || "–")}</span>`;
      const prob = p.prob_positive != null
        ? `<span class="bar" style="width:${Math.round(p.prob_positive * 60)}px;background:${toneColor[tone(p.prob_positive)]}"></span>${pct(p.prob_positive)}` : "–";
      return `<tr data-id="${p.id}" class="${mine.includes(p.id) ? "mine" : ""}">
        <td>${esc(p.name)}</td><td>${esc(p.team || "")}</td><td>${next}</td>
        <td class="num">${pct(p.starter_share_last10)}</td><td class="num">${prob}</td>
        <td class="num">${p.expected_points != null ? (p.expected_points >= 0 ? "+" : "") + p.expected_points.toFixed(1) : "–"}</td></tr>`;
    }).join("");
  }

  // ---------- picking
  function toggle(id, on) {
    mine = mine.filter((x) => x !== id);
    if (on) mine.unshift(id);
    store.set(mine);
    renderCards(); renderTable();
  }

  function setupSearch() {
    const input = $("#search"), list = $("#results");
    let hits = [], active = 0;
    const norm = (s) => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase();
    const show = () => {
      const q = norm(input.value.trim());
      hits = q ? DATA.goalies.filter((p) => norm(`${p.name} ${p.team || ""}`).includes(q)).slice(0, 12) : [];
      active = 0;
      list.hidden = !hits.length;
      list.innerHTML = hits.map((p, i) => `<li role="option" data-id="${p.id}" aria-selected="${i === 0}">
        <span>${esc(p.name)}${mine.includes(p.id) ? " ✓" : ""}</span><span class="team">${esc(p.team || "")}</span></li>`).join("");
    };
    const pick = (id) => { toggle(id, true); input.value = ""; show(); };
    input.addEventListener("input", show);
    input.addEventListener("keydown", (e) => {
      if (!hits.length) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        active = (active + (e.key === "ArrowDown" ? 1 : hits.length - 1)) % hits.length;
        [...list.children].forEach((li, i) => li.setAttribute("aria-selected", i === active));
      } else if (e.key === "Enter") { e.preventDefault(); pick(hits[active].id); }
      else if (e.key === "Escape") { input.value = ""; show(); }
    });
    list.addEventListener("mousedown", (e) => { const li = e.target.closest("li"); if (li) { e.preventDefault(); pick(+li.dataset.id); } });
    input.addEventListener("blur", () => setTimeout(() => (list.hidden = true), 120));
  }

  async function init() {
    try {
      const res = await fetch(`data/predictions.json?t=${Date.now()}`);
      DATA = await res.json();
    } catch (e) {
      $("#cards").innerHTML = `<p class="empty">Couldn't load data/predictions.json — run <code>python -m goalie_predictor update</code>.</p>`;
      return;
    }
    DATA.goalies.forEach((p) => byId.set(p.id, p));
    $("#sample-banner").hidden = !DATA.sample;
    renderScoring(DATA.scoring || {});
    renderMetrics(DATA.metrics, DATA.generated_at);
    mine = (store.get() ?? DATA.default_goalies ?? []).filter((id) => byId.has(id));
    setupSearch();
    document.querySelectorAll("#all th").forEach((th) => th.addEventListener("click", () => {
      sort = { k: th.dataset.k, asc: sort.k === th.dataset.k ? !sort.asc : ["name", "team", "opp"].includes(th.dataset.k) };
      renderTable();
    }));
    $("#all tbody").addEventListener("click", (e) => {
      const tr = e.target.closest("tr"); if (!tr) return;
      const id = +tr.dataset.id; toggle(id, !mine.includes(id));
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    renderCards(); renderTable();
  }
  init();
})();
