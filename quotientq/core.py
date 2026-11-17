"""Finite synthetic market-making MDP, sampled Q-learning, frozen evaluation."""

import math
from copy import deepcopy

HORIZON = 12
EPISODES = (2000, 12000)
TEST_EPISODES = 512
DEFAULT = {"budget": 0, "stress": 0}
ACTIONS = ("Tight", "Wide", "Inventory skew")
POLICIES = ("Learned", "Fixed skew", "Model oracle")


def settings():
    return deepcopy(DEFAULT)


def validate(c):
    if type(c) is not dict or set(c) != set(DEFAULT):
        raise ValueError("settings require exactly budget and stress")
    if any(type(v) is not int or v not in (0, 1) for v in c.values()):
        raise ValueError("settings must be built-in integers 0 or 1")


class Random:
    def __init__(self, seed):
        self.state = seed

    def draw(self):
        self.state = (1664525 * self.state + 1013904223) % 2**32
        return (self.state + 0.5) / 2**32


def index(h, q, signal):
    return (h - 1) * 10 + (q + 2) * 2 + (signal + 1) // 2


def greedy(values):
    maximum = max(values)
    return next(i for i, v in enumerate(values) if v >= maximum - 1e-12)


def quote(q, action):
    if action == 0:
        return (0.5, 0.5)
    if action == 1:
        return (1.5, 1.5)
    if q > 0:
        return (1.5, 0.5)
    if q < 0:
        return (0.5, 1.5)
    return (1.0, 1.0)


def probabilities(q, action, move, stress):
    bid, ask = quote(q, action)
    adverse, benign = (2.5, 0.2) if stress else (1.5, 0.5)
    pb = 0.0 if q == 2 else 0.4 / (1 + bid) * (adverse if move < 0 else benign)
    ps = 0.0 if q == -2 else 0.4 / (1 + ask) * (adverse if move > 0 else benign)
    return pb, ps


def outcome(h, q, signal, action, move, buy, sell, next_signal):
    bid, ask = quote(q, action)
    after = q + buy - sell
    capture = buy * bid + sell * ask
    carry = q * move
    markout = (buy - sell) * move
    fees = 0.03 * (buy + sell)
    penalty = 0.08 * after**2
    exit_cost = 0.25 * abs(after) if h == 1 else 0.0
    reward = capture + carry + markout - fees - penalty - exit_cost
    return {
        "h": h,
        "q": q,
        "signal": signal,
        "action": action,
        "bid_distance": bid,
        "ask_distance": ask,
        "move": move,
        "buy": buy,
        "sell": sell,
        "next_q": after,
        "next_signal": next_signal,
        "capture": capture,
        "carry": carry,
        "markout": markout,
        "fees": fees,
        "penalty": penalty,
        "exit_cost": exit_cost,
        "reward": reward,
    }


def sample(h, q, signal, action, stress, uniforms):
    um, ub, us, un = uniforms
    move = 1 if um < (0.65 if signal == 1 else 0.35) else -1
    pb, ps = probabilities(q, action, move, stress)
    buy, sell = int(ub < pb), int(us < ps)
    next_signal = signal if un < 0.7 else -signal
    return {
        **outcome(h, q, signal, action, move, buy, sell, next_signal),
        "p_buy": pb,
        "p_sell": ps,
        "uniforms": list(uniforms),
    }


def transitions(h, q, signal, action, stress):
    rows = []
    up = 0.65 if signal == 1 else 0.35
    for move in (-1, 1):
        pm = up if move == 1 else 1 - up
        pb, ps = probabilities(q, action, move, stress)
        for buy in (0, 1):
            for sell in (0, 1):
                for ns in (-1, 1):
                    p = (
                        pm
                        * (pb if buy else 1 - pb)
                        * (ps if sell else 1 - ps)
                        * (0.7 if ns == signal else 0.3)
                    )
                    if p > 0:
                        rows.append(
                            {
                                **outcome(h, q, signal, action, move, buy, sell, ns),
                                "probability": p,
                            }
                        )
    return rows


def train(episodes):
    rng = Random(61237)
    table, visits = [[0.0] * 3 for _ in range(120)], [[0] * 3 for _ in range(120)]
    history, running = [], 0.0
    first_update = None
    for episode in range(episodes):
        q, signal = 0, -1 if rng.draw() < 0.5 else 1
        total = 0.0
        for h in range(HORIZON, 0, -1):
            state = index(h, q, signal)
            explore, choice = rng.draw(), rng.draw()
            action = int(choice * 3) if explore < 0.2 else greedy(table[state])
            row = sample(h, q, signal, action, 0, [rng.draw() for _ in range(4)])
            continuation = (
                max(table[index(h - 1, row["next_q"], row["next_signal"])])
                if h > 1
                else 0.0
            )
            target = row["reward"] + continuation
            visits[state][action] += 1
            alpha = 1 / (1 + 0.05 * visits[state][action])
            old = table[state][action]
            table[state][action] += alpha * (target - old)
            if first_update is None:
                first_update = {
                    "state": state,
                    "action": action,
                    "old": old,
                    "target": target,
                    "alpha": alpha,
                    "new": table[state][action],
                    "transition": row,
                }
            total += row["reward"]
            q, signal = row["next_q"], row["next_signal"]
        running += total
        if (episode + 1) % 200 == 0:
            history.append(
                {"episodes": episode + 1, "mean_exploring_reward": running / 200}
            )
            running = 0.0
    return {
        "q_values": table,
        "visits": visits,
        "policy": [greedy(v) for v in table],
        "history": history,
        "first_update": first_update,
        "episodes": episodes,
        "visited_entries": sum(v > 0 for row in visits for v in row),
    }


def solve(stress, policy=None):
    values = [0.0] * 120
    action_values = [[0.0] * 3 for _ in range(120)]
    selected = [0] * 120
    for h in range(1, HORIZON + 1):
        for q in range(-2, 3):
            for signal in (-1, 1):
                state = index(h, q, signal)
                for action in range(3):
                    total = 0.0
                    for row in transitions(h, q, signal, action, stress):
                        future = (
                            values[index(h - 1, row["next_q"], row["next_signal"])]
                            if h > 1
                            else 0.0
                        )
                        total += row["probability"] * (row["reward"] + future)
                    action_values[state][action] = total
                selected[state] = (
                    greedy(action_values[state]) if policy is None else policy[state]
                )
                values[state] = action_values[state][selected[state]]
    initial = (values[index(12, 0, -1)] + values[index(12, 0, 1)]) / 2
    return {
        "values": values,
        "action_values": action_values,
        "policy": selected,
        "initial_value": initial,
    }


def tape(count):
    rng = Random(99881)
    rows = []
    for _ in range(count):
        signal = -1 if rng.draw() < 0.5 else 1
        rows.append(
            {
                "initial_signal": signal,
                "draws": [[rng.draw() for _ in range(4)] for _ in range(HORIZON)],
            }
        )
    return rows


def evaluate(policies, stress, paths):
    results = []
    for name, policy in zip(POLICIES, policies):
        episodes = []
        example = []
        for ep, path in enumerate(paths):
            q, signal, mid, cash, penalty = 0, path["initial_signal"], 100.0, 0.0, 0.0
            fills, inventory_squares = 0, 0
            total_reward = 0.0
            for step, uniforms in enumerate(path["draws"]):
                h = HORIZON - step
                action = policy[index(h, q, signal)]
                row = sample(h, q, signal, action, stress, uniforms)
                cash += (
                    row["sell"] * (mid + row["ask_distance"])
                    - row["buy"] * (mid - row["bid_distance"])
                    - row["fees"]
                )
                next_mid = mid + row["move"]
                marked = cash + row["next_q"] * next_mid
                penalty += row["penalty"]
                fills += row["buy"] + row["sell"]
                inventory_squares += row["next_q"] ** 2
                total_reward += row["reward"]
                if ep == 0:
                    example.append(
                        {
                            **row,
                            "step": step + 1,
                            "mid": mid,
                            "next_mid": next_mid,
                            "cash": cash,
                            "marked": marked,
                            "cumulative_penalty": penalty,
                            "cumulative_reward": total_reward,
                        }
                    )
                q, signal, mid = row["next_q"], row["next_signal"], next_mid
            pnl = cash + q * mid - 0.25 * abs(q)
            episodes.append(
                {
                    "id": ep + 1,
                    "pnl_after_exit": pnl,
                    "inventory_penalty": penalty,
                    "objective": pnl - penalty,
                    "reward_sum": total_reward,
                    "reconciliation_gap": total_reward - (pnl - penalty),
                    "fills": fills,
                    "inventory_squares": inventory_squares,
                    "terminal_q_before_exit": q,
                }
            )
        n = len(episodes)
        results.append(
            {
                "name": name,
                "episodes": episodes,
                "example": example,
                "mean_objective": sum(r["objective"] for r in episodes) / n,
                "mean_pnl": sum(r["pnl_after_exit"] for r in episodes) / n,
                "mean_penalty": sum(r["inventory_penalty"] for r in episodes) / n,
                "rms_inventory": math.sqrt(
                    sum(r["inventory_squares"] for r in episodes) / (n * HORIZON)
                ),
                "mean_fills": sum(r["fills"] for r in episodes) / n,
            }
        )
    return results


def analyze(value=None):
    c = settings() if value is None else value
    validate(c)
    c = deepcopy(c)
    learned = train(EPISODES[c["budget"]])
    baseline = [2] * 120
    oracle = solve(c["stress"])
    learned_value = solve(c["stress"], learned["policy"])
    baseline_value = solve(c["stress"], baseline)
    paths = tape(TEST_EPISODES)
    evaluations = evaluate(
        [learned["policy"], baseline, oracle["policy"]], c["stress"], paths
    )
    for result, exact in zip(evaluations, [learned_value, baseline_value, oracle]):
        result["exact_expected_objective"] = exact["initial_value"]
    states = []
    for h in range(1, 13):
        for q in range(-2, 3):
            for signal in (-1, 1):
                i = index(h, q, signal)
                reachable = abs(q) <= min(2, 12 - h)
                states.append(
                    {
                        "id": i,
                        "remaining": h,
                        "inventory": q,
                        "signal": signal,
                        "reachable_from_start": reachable,
                        "learned_action": learned["policy"][i],
                        "oracle_action": oracle["policy"][i],
                        "q_values": learned["q_values"][i],
                        "visits": learned["visits"][i],
                        "oracle_value": oracle["values"][i],
                        "learned_policy_value": learned_value["values"][i],
                    }
                )
    return {
        "settings": c,
        "training": learned,
        "states": states,
        "oracle": oracle,
        "evaluations": evaluations,
        "exact_optimality_gap": oracle["initial_value"]
        - learned_value["initial_value"],
        "test_episodes": TEST_EPISODES,
        "test_tape": paths,
    }
