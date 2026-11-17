import json
import math
import subprocess
import sys
import unittest
from unittest.mock import patch
from quotientq.core import (
    analyze,
    settings,
    train,
    solve,
    index,
    quote,
    probabilities,
    sample,
    transitions,
    greedy,
)


def independent_step(h, q, s, a, stress, u):
    distances = (
        (0.5, 0.5),
        (1.5, 1.5),
        (1.5, 0.5) if q > 0 else (0.5, 1.5) if q < 0 else (1.0, 1.0),
    )[a]
    d = 1 if u[0] < (0.5 + 0.15 * s) else -1
    m = (2.5, 0.2) if stress else (1.5, 0.5)
    pb = 0 if q >= 2 else (0.4 / (1 + distances[0])) * m[int(d > 0)]
    ps = 0 if q <= -2 else (0.4 / (1 + distances[1])) * m[int(d < 0)]
    b = int(u[1] < pb)
    sell = int(u[2] < ps)
    after = q + b - sell
    reward = (
        b * distances[0]
        + sell * distances[1]
        + after * d
        - 0.03 * (b + sell)
        - 0.08 * after * after
        - (0.25 * abs(after) if h == 1 else 0)
    )
    return after, s if u[3] < 0.7 else -s, reward, b, sell, d, distances


def independent_train(count):
    seed = 61237

    def uniform():
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return (seed + 0.5) / 4294967296

    values = {
        (h, q, s): [0.0, 0.0, 0.0]
        for h in range(1, 13)
        for q in range(-2, 3)
        for s in [-1, 1]
    }
    visits = {key: [0, 0, 0] for key in values}
    for _ in range(count):
        q = 0
        s = -1 if uniform() < 0.5 else 1
        for h in range(12, 0, -1):
            key = (h, q, s)
            u0, u1 = uniform(), uniform()
            v = values[key]
            a = (
                int(u1 * 3)
                if u0 < 0.2
                else next(j for j in range(3) if v[j] >= max(v) - 1e-12)
            )
            nq, ns, r, *_ = independent_step(
                h, q, s, a, 0, [uniform() for _ in range(4)]
            )
            target = r + (max(values[(h - 1, nq, ns)]) if h > 1 else 0)
            visits[key][a] += 1
            rate = 1 / (1 + 0.05 * visits[key][a])
            v[a] = (1 - rate) * v[a] + rate * target
            q, s = nq, ns
    return values, visits


class Core(unittest.TestCase):
    def test_01_full_independent_training(self):
        for n in [2000, 12000]:
            actual = train(n)
            expected, visits = independent_train(n)
            for key, v in expected.items():
                i = index(*key)
                self.assertEqual(actual["visits"][i], visits[key])
                for a in range(3):
                    self.assertAlmostEqual(actual["q_values"][i][a], v[a], places=11)
                self.assertEqual(
                    actual["policy"][i],
                    next(a for a in range(3) if v[a] >= max(v) - 1e-12),
                )
            self.assertEqual(sum(sum(row) for row in actual["visits"]), 12 * n)

    def test_02_every_transition_probability_reward(self):
        for stress in [0, 1]:
            for h in range(1, 13):
                for q in range(-2, 3):
                    for s in [-1, 1]:
                        for a in range(3):
                            rows = transitions(h, q, s, a, stress)
                            self.assertAlmostEqual(
                                sum(v["probability"] for v in rows), 1, places=14
                            )
                            for r in rows:
                                self.assertLessEqual(abs(r["next_q"]), 2)
                                self.assertFalse(q == 2 and r["buy"])
                                self.assertFalse(q == -2 and r["sell"])
                                self.assertAlmostEqual(
                                    r["reward"],
                                    r["capture"]
                                    + r["next_q"] * r["move"]
                                    - r["fees"]
                                    - r["penalty"]
                                    - r["exit_cost"],
                                )
                                self.assertEqual(
                                    r["exit_cost"],
                                    0.25 * abs(r["next_q"]) if h == 1 else 0,
                                )
                            for move in [-1, 1]:
                                b, p = probabilities(q, a, move, stress)
                                self.assertGreaterEqual(b, 0)
                                self.assertLess(b, 1)
                                self.assertGreaterEqual(p, 0)
                                self.assertLess(p, 1)

    def test_03_independent_bellman_enumeration(self):
        # Separate Cartesian-product algebra, no production transition/outcome/quote helper.
        import itertools

        for stress in [0, 1]:
            vals = {}
            chosen = {}
            for h in range(1, 13):
                for q in range(-2, 3):
                    for s in [-1, 1]:
                        rewards = []
                        for a in range(3):
                            db, ds = [
                                (0.5, 0.5),
                                (1.5, 1.5),
                                (
                                    (1.5, 0.5)
                                    if q > 0
                                    else (0.5, 1.5) if q < 0 else (1.0, 1.0)
                                ),
                            ][a]
                            total = 0
                            for d, b, k, ns in itertools.product(
                                [-1, 1], [0, 1], [0, 1], [-1, 1]
                            ):
                                adv, ben = (2.5, 0.2) if stress else (1.5, 0.5)
                                pb = (
                                    0
                                    if q == 2
                                    else 0.4 / (1 + db) * (adv if d == -1 else ben)
                                )
                                ps = (
                                    0
                                    if q == -2
                                    else 0.4 / (1 + ds) * (adv if d == 1 else ben)
                                )
                                prob = (
                                    ((0.5 + 0.15 * s) if d == 1 else (0.5 - 0.15 * s))
                                    * (pb if b else 1 - pb)
                                    * (ps if k else 1 - ps)
                                    * (0.7 if ns == s else 0.3)
                                )
                                if not prob:
                                    continue
                                nq = q + b - k
                                r = (
                                    db * b
                                    + ds * k
                                    + nq * d
                                    - 0.03 * (b + k)
                                    - 0.08 * nq * nq
                                    - (0.25 * abs(nq) if h == 1 else 0)
                                )
                                total += prob * (
                                    r + (vals[(h - 1, nq, ns)] if h > 1 else 0)
                                )
                            rewards.append(total)
                            self.assertAlmostEqual(
                                total,
                                solve_cache[stress]["action_values"][index(h, q, s)][a],
                                places=12,
                            )
                        a = next(
                            i
                            for i, v in enumerate(rewards)
                            if v >= max(rewards) - 1e-12
                        )
                        vals[h, q, s] = rewards[a]
                        chosen[h, q, s] = a
            actual = solve_cache[stress]
            for key, v in vals.items():
                self.assertAlmostEqual(v, actual["values"][index(*key)], places=12)
                self.assertEqual(chosen[key], actual["policy"][index(*key)])

    def test_04_frozen_training_and_paths(self):
        a = analyze({"budget": 0, "stress": 0})
        b = analyze({"budget": 0, "stress": 1})
        self.assertEqual(a["training"], b["training"])
        self.assertEqual(a["test_tape"], b["test_tape"])
        small = train(2000)
        big = train(12000)
        self.assertEqual(small["history"], big["history"][:10])
        for v in a["states"]:
            if not v["reachable_from_start"]:
                self.assertEqual(v["visits"], [0] * 3)
        self.assertEqual(big["visited_entries"], 324)
        original = train(2000)
        with patch(
            "quotientq.core.solve", side_effect=RuntimeError("oracle inaccessible")
        ):
            self.assertEqual(train(2000), original)

    def test_05_replay_every_test_episode(self):
        for stress in [0, 1]:
            result = analyze({"budget": 0, "stress": stress})
            policies = [
                result["training"]["policy"],
                [2] * 120,
                result["oracle"]["policy"],
            ]
            for p, policy in enumerate(policies):
                for path, reported in zip(
                    result["test_tape"], result["evaluations"][p]["episodes"]
                ):
                    q, s, cash, mid, penalty, total, inv, fills = (
                        0,
                        path["initial_signal"],
                        0.0,
                        100.0,
                        0.0,
                        0.0,
                        0,
                        0,
                    )
                    for step, u in enumerate(path["draws"]):
                        h = 12 - step
                        a = policy[index(h, q, s)]
                        nq, ns, r, b, k, d, dist = independent_step(
                            h, q, s, a, stress, u
                        )
                        cash += (
                            k * (mid + dist[1]) - b * (mid - dist[0]) - 0.03 * (b + k)
                        )
                        mid += d
                        penalty += 0.08 * nq * nq
                        total += r
                        inv += nq * nq
                        fills += b + k
                        q, s = nq, ns
                    pnl = cash + q * mid - 0.25 * abs(q)
                    self.assertAlmostEqual(pnl, reported["pnl_after_exit"], places=11)
                    self.assertAlmostEqual(
                        penalty, reported["inventory_penalty"], places=11
                    )
                    self.assertAlmostEqual(total, reported["objective"], places=11)
                    self.assertAlmostEqual(pnl - penalty, total, places=11)
                    self.assertEqual(inv, reported["inventory_squares"])
                    self.assertEqual(fills, reported["fills"])
                    self.assertEqual(q, reported["terminal_q_before_exit"])
                    self.assertLess(abs(reported["reconciliation_gap"]), 1e-11)

    def test_06_expected_policy_bounds_and_objective(self):
        for c in [{"budget": b, "stress": s} for b in [0, 1] for s in [0, 1]]:
            r = analyze(c)
            self.assertGreaterEqual(r["exact_optimality_gap"], -1e-12)
            for p in r["evaluations"]:
                self.assertAlmostEqual(
                    p["mean_objective"], p["mean_pnl"] - p["mean_penalty"], places=12
                )
                self.assertLessEqual(
                    p["exact_expected_objective"], r["oracle"]["initial_value"] + 1e-12
                )
                self.assertEqual(len(p["example"]), 12)
            for state in r["states"]:
                self.assertLessEqual(
                    state["learned_policy_value"], state["oracle_value"] + 1e-12
                )

    def test_07_terminal_target_and_hand_fills(self):
        # At last step: buy at99.5, mid falls1, inventory1, fee.03, penalty.08, exit.25.
        row = sample(1, 0, 1, 0, 0, [0.99, 0, 0.99, 0])
        self.assertEqual(row["buy"], 1)
        self.assertEqual(row["sell"], 0)
        self.assertAlmostEqual(row["reward"], -0.86)
        blocked = sample(1, 2, 1, 0, 0, [0.99, 0, 0, 0])
        self.assertEqual(blocked["buy"], 0)
        self.assertEqual(blocked["sell"], 1)
        self.assertEqual(blocked["next_q"], 1)
        first = train(1)["first_update"]
        self.assertEqual(first["target"], first["transition"]["reward"])
        self.assertAlmostEqual(first["new"], first["alpha"] * first["target"])
        self.assertEqual(greedy([1.0, 1.0 + 5e-13, 0]), 0)

    def test_08_config_cli_detached(self):
        for c in [
            {},
            [],
            False,
            {"budget": False, "stress": 0},
            {"budget": 0, "stress": 2},
            {"budget": 0, "stress": 0, "x": 1},
        ]:
            with self.assertRaises(ValueError):
                analyze(c)
        c = settings()
        r = analyze(c)
        r["settings"]["stress"] = 1
        self.assertEqual(c, settings())
        r["training"]["policy"][0] = 999
        self.assertNotEqual(analyze()["training"]["policy"][0], 999)
        p = subprocess.run(
            [sys.executable, "-m", "quotientq", "--stress", "1"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(p.stdout)["settings"], {"budget": 0, "stress": 1})
        p = subprocess.run(
            [sys.executable, "-m", "quotientq", "--budget", "2"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 2)


solve_cache = {s: solve(s) for s in [0, 1]}
if __name__ == "__main__":
    unittest.main()
