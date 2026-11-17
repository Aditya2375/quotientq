import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { mkdir, readFile } from "node:fs/promises";
import assert from "node:assert/strict";
import { analyze, settings, ACTIONS, index } from "../docs/core.js";
const base = process.env.BASE_URL || "http://127.0.0.1:8385/";
let server;
if (!process.env.BASE_URL) {
  server = spawn(
    "python3",
    ["-m", "http.server", "8385", "--directory", "docs", "--bind", "127.0.0.1"],
    { stdio: "ignore" },
  );
  for (let i = 0; i < 50; i++) {
    try {
      if ((await fetch(base)).ok) break;
    } catch {}
    await new Promise((r) => setTimeout(r, 100));
  }
}
await mkdir("/tmp/quotientq-shots", { recursive: true });
const browser = await chromium.launch({ headless: true });
let checks = 0;
const check = (v, m) => {
    checks++;
    assert(v, m);
  },
  fmt = (v, d = 3) => v.toFixed(d),
  configs = [
    { budget: 0, stress: 0 },
    { budget: 0, stress: 1 },
    { budget: 1, stress: 0 },
    { budget: 1, stress: 1 },
  ],
  states = configs.map(analyze);
try {
  for (const width of [1440, 1024, 768, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } }),
      errors = [],
      failed = [],
      third = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("requestfailed", (r) => failed.push(r.url()));
    page.on("request", (r) => {
      if (!r.url().startsWith(base) && !r.url().startsWith("blob:"))
        third.push(r.url());
    });
    await page.goto(base);
    await page
      .locator("#status")
      .filter({ hasText: "Policy frozen" })
      .waitFor();
    async function table(id) {
      return page
        .locator("#" + id + " tr")
        .evaluateAll((rows) =>
          rows.map((tr) =>
            [...tr.querySelectorAll("td")].map((td) => td.textContent),
          ),
        );
    }
    async function audit(id, r) {
      await page.locator(`.policy-map button[data-id="${id}"]`).click();
      const s = r.states[id];
      check(
        (await page.locator("#chosen").textContent()) ===
          ACTIONS[s.learned_action],
        "selected action",
      );
      check(
        (await page.locator("#oracle-action").textContent()) ===
          ACTIONS[s.oracle_action],
        "selected oracle",
      );
      check(
        JSON.stringify(await table("state-values")) ===
          JSON.stringify(
            ACTIONS.map((a, i) => [a, fmt(s.q_values[i]), String(s.visits[i])]),
          ),
        "selected values+visits",
      );
      check(
        (await page
          .locator('.policy-map button[aria-pressed="true"]')
          .count()) === 1,
        "one selected",
      );
    }
    async function verify(r) {
      for (const [id, v] of [
        ["exact", fmt(r.evaluations[0].exact_expected_objective)],
        ["gap", fmt(r.exact_optimality_gap)],
        ["coverage", `${r.training.visited_entries} / 360`],
      ])
        check(
          (await page.locator("#" + id).textContent()) === v,
          `${width}:${id}`,
        );
      check(
        JSON.stringify(await table("scores")) ===
          JSON.stringify(
            r.evaluations.map((p) => [
              p.name,
              fmt(p.exact_expected_objective),
              fmt(p.mean_objective),
              fmt(p.mean_pnl),
              fmt(p.mean_penalty),
              fmt(p.rms_inventory),
              fmt(p.mean_fills),
            ]),
          ),
        "all score fields",
      );
      check(
        JSON.stringify(await table("state-rows")) ===
          JSON.stringify(
            r.states.map((s) => [
              String(s.remaining),
              String(s.inventory),
              String(s.signal),
              s.reachable_from_start ? "yes" : "no",
              ACTIONS[s.learned_action],
              ACTIONS[s.oracle_action],
              ...s.q_values.map((v) => fmt(v, 5)),
              ...s.visits.map(String),
            ]),
          ),
        "all120 state rows",
      );
      check(
        JSON.stringify(await table("episode-rows")) ===
          JSON.stringify(
            r.evaluations[0].episodes.map((e, i) => [
              String(e.id),
              ...r.evaluations.flatMap((p) => {
                const v = p.episodes[i];
                return [
                  fmt(v.pnl_after_exit),
                  fmt(v.inventory_penalty),
                  fmt(v.objective),
                ];
              }),
            ]),
          ),
        "all512testrows",
      );
      check(
        JSON.stringify(await table("example-rows")) ===
          JSON.stringify(
            r.evaluations.flatMap((p) =>
              p.example.map((v) => [
                p.name,
                String(v.step),
                String(v.q),
                String(v.signal),
                ACTIONS[v.action],
                fmt(v.bid_distance, 1),
                fmt(v.ask_distance, 1),
                String(v.buy),
                String(v.sell),
                String(v.move),
                String(v.next_q),
                fmt(v.cash),
                fmt(v.marked),
                fmt(v.penalty),
                fmt(v.exit_cost),
                fmt(v.reward),
              ]),
            ),
          ),
        "all36eventrows",
      );
      const cells = await page.locator(".policy-map button").evaluateAll((bs) =>
        bs.map((b) => ({
          id: Number(b.dataset.id),
          text: b.textContent,
          off: b.classList.contains("unreachable"),
          mismatch: b.classList.contains("mismatch"),
        })),
      );
      check(cells.length === 120, "120buttons");
      for (const b of cells) {
        const s = r.states[b.id];
        check(b.text === ["T", "W", "S"][s.learned_action], "policy letter");
        check(b.off === !s.reachable_from_start, "reachability");
        check(
          b.mismatch === (s.learned_action !== s.oracle_action),
          "mismatch marker",
        );
      }
      check(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        "page containment",
      );
      for (const id of ["learning", "paired"])
        check(
          await page.locator("#" + id).evaluate(
            (c) =>
              c.width >= Math.floor(c.getBoundingClientRect().width) &&
              c
                .getContext("2d")
                .getImageData(0, 0, c.width, c.height)
                .data.some((v, i) => i % 4 === 3 && v > 0),
          ),
          id + "canvas",
        );
    }
    await verify(states[0]);
    if (width === 1440)
      await page.screenshot({
        path: "/tmp/quotientq-shots/desktop.png",
        fullPage: true,
      });
    for (let i = 0; i < 4; i++) {
      const c = configs[i];
      await page.locator("#budget").selectOption(String(c.budget));
      await page.locator("#stress").selectOption(String(c.stress));
      await verify(states[i]);
      for (const id of [index(12, 0, 1), index(1, 2, -1), index(12, -2, -1)])
        await audit(id, states[i]);
      if (width === 320 && i === 3)
        await page.screenshot({
          path: "/tmp/quotientq-shots/mobile-stress.png",
          fullPage: true,
        });
    }
    await page.locator("#reset").click();
    await verify(states[0]);
    check((await page.locator("#budget").inputValue()) === "0", "reset budget");
    check((await page.locator("#stress").inputValue()) === "0", "reset stress");
    for (const detail of await page.locator("details").all())
      await detail.locator("summary").click();
    if (width === 320)
      await page
        .locator(".ledgers")
        .screenshot({ path: "/tmp/quotientq-shots/mobile-ledgers.png" });
    const dl = page.waitForEvent("download");
    await page.locator("#download").click();
    const download = await dl;
    check(
      JSON.stringify(
        JSON.parse(await readFile(await download.path(), "utf8")),
      ) === JSON.stringify(states[0]),
      "full export",
    );
    await page.locator("#stress").evaluate((e) => {
      const o = document.createElement("option");
      o.value = "9";
      e.append(o);
      e.value = "9";
      e.dispatchEvent(new Event("change", { bubbles: true }));
    });
    check(
      (await page.locator("#status").textContent()).startsWith(
        "Invalid choice rejected",
      ),
      "invalid guard",
    );
    check(
      (await page.locator("#stress").inputValue()) === "0",
      "invalid restore",
    );
    check(
      (await page.locator("#exact").textContent()) ===
        fmt(states[0].evaluations[0].exact_expected_objective),
      "invalid retains",
    );
    check((await page.context().cookies()).length === 0, "no cookies");
    check(
      await page.evaluate(
        () => localStorage.length === 0 && sessionStorage.length === 0,
      ),
      "no storage",
    );
    check(
      await page.evaluate(async () =>
        navigator.serviceWorker
          ? (await navigator.serviceWorker.getRegistrations()).length === 0
          : true,
      ),
      "noSW",
    );
    check(errors.length === 0, JSON.stringify(errors));
    check(failed.length === 0, JSON.stringify(failed));
    check(third.length === 0, JSON.stringify(third));
    await page.close();
  }
  console.log(
    `${checks} Chromium assertions; 5 widths / 4 configs / all 120 policy states + 512 test episodes + 36 events / selection / reset / export / invalid / privacy PASS`,
  );
} finally {
  await browser.close();
  if (server) {
    server.kill();
    await new Promise((r) => server.once("close", r));
  }
}
