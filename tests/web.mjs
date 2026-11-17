import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { analyze, settings } from "../docs/core.js";
let assertions = 0;
const check = (v, m) => {
  assertions++;
  assert(v, m);
};
function compare(a, b, path = "") {
  if (typeof a === "number") {
    check(Number.isFinite(a) && Number.isFinite(b), path + " finite");
    check(
      Math.abs(a - b) < 2e-13 + Math.abs(a) * 3e-11,
      `${path}: ${a} != ${b}`,
    );
  } else if (a === null || typeof a !== "object") check(a === b, path);
  else {
    check(
      JSON.stringify(Object.keys(a)) === JSON.stringify(Object.keys(b)),
      path + " keys",
    );
    for (const k of Object.keys(a)) compare(a[k], b[k], path + "." + k);
  }
}
const configs = [];
for (const budget of [0, 1])
  for (const stress of [0, 1]) configs.push({ budget, stress });
const run = spawnSync(
  "python3",
  [
    "-c",
    "import json,sys;from quotientq.core import analyze;print(json.dumps([analyze(c)for c in json.load(sys.stdin)],allow_nan=False))",
  ],
  {
    input: JSON.stringify(configs),
    encoding: "utf8",
    maxBuffer: 128 * 1024 * 1024,
  },
);
assert.equal(run.status, 0, run.stderr);
const expected = JSON.parse(run.stdout);
configs.forEach((c, i) => compare(analyze(c), expected[i], "full" + i));
for (const budget of [0, 1]) {
  const a = analyze({ budget, stress: 0 }),
    b = analyze({ budget, stress: 1 });
  compare(a.training, b.training);
  compare(a.test_tape, b.test_tape);
  check(a.exact_optimality_gap >= -1e-10, "oracle expected bound");
  check(b.exact_optimality_gap >= -1e-10, "oracle shifted bound");
}
for (const bad of [
  null,
  {},
  [],
  { budget: true, stress: 0 },
  { budget: 2, stress: 0 },
  { budget: 0, stress: 0, extra: 1 },
]) {
  assert.throws(() => analyze(bad));
  assertions++;
}
console.log(
  `${assertions} JS assertions; 4 complete Python states, frozen training under evaluation shift and exact expected bounds PASS`,
);
