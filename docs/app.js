import { analyze, settings, validate, ACTIONS, index } from "./core.js";
const $ = (id) => document.getElementById(id),
  put = (id, v) => {
    $(id).textContent = v;
  },
  fmt = (v, d = 3) => v.toFixed(d),
  letters = ["T", "W", "S"],
  classes = ["tight", "wide", "skew"];
let state = settings(),
  result,
  selected = index(12, 0, 1);
function rows(id, data) {
  const fragment = document.createDocumentFragment();
  for (const cells of data) {
    const tr = document.createElement("tr");
    for (const v of cells) {
      const td = document.createElement("td");
      td.textContent = v;
      tr.append(td);
    }
    fragment.append(tr);
  }
  $(id).replaceChildren(fragment);
}
function audit() {
  const s = result.states[selected];
  put(
    "state-title",
    `${s.remaining} steps · q ${s.inventory} · signal ${s.signal > 0 ? "+1" : "−1"}`,
  );
  put(
    "state-note",
    s.reachable_from_start
      ? "Reachable from the zero-inventory start. Values are finite-sample estimates."
      : "Unreachable from the zero-inventory start at this remaining time. No training visit should be inferred.",
  );
  put("chosen", ACTIONS[s.learned_action]);
  put("oracle-action", ACTIONS[s.oracle_action]);
  rows(
    "state-values",
    ACTIONS.map((a, i) => [a, fmt(s.q_values[i]), String(s.visits[i])]),
  );
  for (const b of document.querySelectorAll(".policy-map button")) {
    const active = Number(b.dataset.id) === selected;
    b.classList.toggle("selected", active);
    b.setAttribute("aria-pressed", String(active));
  }
}
function maps() {
  for (const [id, signal] of [
    ["map-negative", -1],
    ["map-positive", 1],
  ]) {
    const frag = document.createDocumentFragment();
    for (const v of ["h/q", -2, -1, 0, 1, 2]) {
      const el = document.createElement("span");
      el.className = "axis";
      el.textContent = v;
      frag.append(el);
    }
    for (let h = 12; h > 0; h--) {
      const el = document.createElement("span");
      el.className = "axis";
      el.textContent = h;
      frag.append(el);
      for (let q = -2; q <= 2; q++) {
        const s = result.states[index(h, q, signal)],
          b = document.createElement("button");
        b.type = "button";
        b.textContent = letters[s.learned_action];
        b.dataset.id = s.id;
        b.classList.add(classes[s.learned_action]);
        if (!s.reachable_from_start) b.classList.add("unreachable");
        if (s.learned_action !== s.oracle_action) b.classList.add("mismatch");
        b.setAttribute(
          "aria-label",
          `${h} steps remaining, inventory ${q}, signal ${signal}: ${ACTIONS[s.learned_action]}; oracle ${ACTIONS[s.oracle_action]}; ${s.reachable_from_start ? "reachable" : "unreachable"}`,
        );
        b.addEventListener("click", () => {
          selected = s.id;
          audit();
        });
        frag.append(b);
      }
    }
    $(id).replaceChildren(frag);
  }
  audit();
}
function canvas(id) {
  const c = $(id),
    w = Math.floor(c.getBoundingClientRect().width),
    h = Math.floor(c.getBoundingClientRect().height),
    r = devicePixelRatio || 1;
  c.width = w * r;
  c.height = h * r;
  const ctx = c.getContext("2d");
  ctx.scale(r, r);
  ctx.font = "10px system-ui";
  return { ctx, w, h };
}
function frame(ctx, w, h, lo, hi, xlo, xhi, xlabel) {
  const l = 40,
    r = 20,
    t = 15,
    b = 36,
    px = (x) => l + ((x - xlo) / (xhi - xlo)) * (w - l - r),
    py = (y) => t + ((hi - y) / (hi - lo)) * (h - t - b);
  ctx.fillStyle = "#716a78";
  ctx.strokeStyle = "#e1dbe2";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(l, py(v));
    ctx.lineTo(w - r, py(v));
    ctx.stroke();
    ctx.fillText(v.toFixed(1), l - 6, py(v) + 3);
  }
  ctx.textAlign = "center";
  ctx.fillText(xlabel, l + (w - l - r) / 2, h - 2);
  return { px, py, t, b };
}
function learning() {
  const { ctx, w, h } = canvas("learning"),
    data = result.training.history,
    values = data.map((v) => v.mean_exploring_reward),
    lo = Math.min(0, ...values) - 0.3,
    hi = Math.max(0, ...values) + 0.3,
    { px, py } = frame(
      ctx,
      w,
      h,
      lo,
      hi,
      0,
      result.training.episodes,
      "Training episodes",
    );
  ctx.strokeStyle = "#75638e";
  ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((v, i) =>
    i
      ? ctx.lineTo(px(v.episodes), py(v.mean_exploring_reward))
      : ctx.moveTo(px(v.episodes), py(v.mean_exploring_reward)),
  );
  ctx.stroke();
  ctx.fillStyle = "#75638e";
  for (const v of data) {
    ctx.beginPath();
    ctx.arc(px(v.episodes), py(v.mean_exploring_reward), 2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#716a78";
  for (const v of [0, result.training.episodes / 2, result.training.episodes])
    ctx.fillText(String(v), px(v), h - 19);
}
function paired() {
  const { ctx, w, h } = canvas("paired"),
    a = result.evaluations[0].episodes,
    b = result.evaluations[1].episodes,
    data = a.map((v, i) => v.objective - b[i].objective),
    lo = Math.min(...data) - 0.1,
    hi = Math.max(...data) + 0.1,
    bins = Array(16).fill(0),
    step = (hi - lo) / bins.length;
  for (const v of data) bins[Math.min(15, Math.floor((v - lo) / step))]++;
  const { px, py, t } = frame(
    ctx,
    w,
    h,
    0,
    Math.max(...bins) * 1.15,
    lo,
    hi,
    "Paired objective difference",
  );
  ctx.fillStyle = "#b8a5c7";
  for (let i = 0; i < 16; i++)
    ctx.fillRect(
      px(lo + i * step) + 1,
      py(bins[i]),
      Math.max(1, px(lo + (i + 1) * step) - px(lo + i * step) - 2),
      py(0) - py(bins[i]),
    );
  ctx.strokeStyle = "#af552c";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(px(0), t);
  ctx.lineTo(px(0), py(0));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#716a78";
  for (let i = 0; i <= 3; i++) {
    const v = lo + ((hi - lo) * i) / 3;
    ctx.fillText(v.toFixed(1), px(v), h - 19);
  }
}
function render() {
  result = analyze(state);
  put(
    "status",
    state.stress
      ? "Training stays fixed. Only the separate test fill law changes; the model oracle is recalculated for that test law."
      : "Policy frozen after training. Evaluation uses a separate seed and no exploration; oracle values never enter learning.",
  );
  put("exact", fmt(result.evaluations[0].exact_expected_objective));
  put("gap", fmt(result.exact_optimality_gap));
  put("coverage", `${result.training.visited_entries} / 360`);
  rows(
    "scores",
    result.evaluations.map((p) => [
      p.name,
      fmt(p.exact_expected_objective),
      fmt(p.mean_objective),
      fmt(p.mean_pnl),
      fmt(p.mean_penalty),
      fmt(p.rms_inventory),
      fmt(p.mean_fills),
    ]),
  );
  rows(
    "state-rows",
    result.states.map((s) => [
      s.remaining,
      s.inventory,
      s.signal,
      s.reachable_from_start ? "yes" : "no",
      ACTIONS[s.learned_action],
      ACTIONS[s.oracle_action],
      ...s.q_values.map((v) => fmt(v, 5)),
      ...s.visits,
    ]),
  );
  rows(
    "episode-rows",
    result.evaluations[0].episodes.map((e, i) => [
      e.id,
      ...result.evaluations.flatMap((p) => {
        const e = p.episodes[i];
        return [
          fmt(e.pnl_after_exit),
          fmt(e.inventory_penalty),
          fmt(e.objective),
        ];
      }),
    ]),
  );
  rows(
    "example-rows",
    result.evaluations.flatMap((p) =>
      p.example.map((r) => [
        p.name,
        r.step,
        r.q,
        r.signal,
        ACTIONS[r.action],
        fmt(r.bid_distance, 1),
        fmt(r.ask_distance, 1),
        r.buy,
        r.sell,
        r.move,
        r.next_q,
        fmt(r.cash),
        fmt(r.marked),
        fmt(r.penalty),
        fmt(r.exit_cost),
        fmt(r.reward),
      ]),
    ),
  );
  maps();
  learning();
  paired();
}
for (const key of ["budget", "stress"])
  $(key).addEventListener("change", (event) => {
    const next = { ...state, [key]: Number(event.target.value) };
    try {
      validate(next);
      state = next;
      render();
    } catch {
      event.target.value = String(state[key]);
      put("status", "Invalid choice rejected. Previous experiment unchanged.");
    }
  });
$("reset").addEventListener("click", () => {
  state = settings();
  selected = index(12, 0, 1);
  for (const key of ["budget", "stress"]) $(key).value = String(state[key]);
  render();
});
$("download").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    }),
    url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = "quotientq-experiment.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
let pending = false;
window.addEventListener("resize", () => {
  if (!pending) {
    pending = true;
    requestAnimationFrame(() => {
      learning();
      paired();
      pending = false;
    });
  }
});
render();
