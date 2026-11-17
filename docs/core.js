export const HORIZON = 12,
  EPISODES = [2000, 12000],
  TEST_EPISODES = 512,
  ACTIONS = ["Tight", "Wide", "Inventory skew"],
  POLICIES = ["Learned", "Fixed skew", "Model oracle"];
export const settings = () => ({ budget: 0, stress: 0 });
export function validate(c) {
  if (
    !c ||
    Object.getPrototypeOf(c) !== Object.prototype ||
    Object.keys(c).sort().join(",") !== "budget,stress"
  )
    throw new Error("settings require exactly budget and stress");
  if (Object.values(c).some((v) => !Number.isInteger(v) || v < 0 || v > 1))
    throw new Error("settings must be integers 0 or 1");
}
export class Random {
  constructor(seed) {
    this.state = seed;
  }
  draw() {
    this.state = (Math.imul(1664525, this.state) + 1013904223) >>> 0;
    return (this.state + 0.5) / 4294967296;
  }
}
export const index = (h, q, signal) =>
  (h - 1) * 10 + (q + 2) * 2 + (signal + 1) / 2;
export function greedy(values) {
  const maximum = Math.max(...values);
  return values.findIndex((v) => v >= maximum - 1e-12);
}
export function quote(q, a) {
  if (a === 0) return [0.5, 0.5];
  if (a === 1) return [1.5, 1.5];
  if (q > 0) return [1.5, 0.5];
  if (q < 0) return [0.5, 1.5];
  return [1, 1];
}
export function probabilities(q, a, move, stress) {
  const [bid, ask] = quote(q, a),
    [adverse, benign] = stress ? [2.5, 0.2] : [1.5, 0.5];
  return [
    q === 2 ? 0 : (0.4 / (1 + bid)) * (move < 0 ? adverse : benign),
    q === -2 ? 0 : (0.4 / (1 + ask)) * (move > 0 ? adverse : benign),
  ];
}
export function outcome(h, q, signal, action, move, buy, sell, next_signal) {
  const [bid, ask] = quote(q, action),
    after = q + buy - sell,
    capture = buy * bid + sell * ask,
    carry = q * move,
    markout = (buy - sell) * move,
    fees = 0.03 * (buy + sell),
    penalty = 0.08 * after ** 2,
    exit_cost = h === 1 ? 0.25 * Math.abs(after) : 0,
    reward = capture + carry + markout - fees - penalty - exit_cost;
  return {
    h,
    q,
    signal,
    action,
    bid_distance: bid,
    ask_distance: ask,
    move,
    buy,
    sell,
    next_q: after,
    next_signal,
    capture,
    carry,
    markout,
    fees,
    penalty,
    exit_cost,
    reward,
  };
}
export function sample(h, q, signal, action, stress, uniforms) {
  const [um, ub, us, un] = uniforms,
    move = um < (signal === 1 ? 0.65 : 0.35) ? 1 : -1,
    [pb, ps] = probabilities(q, action, move, stress),
    buy = +(ub < pb),
    sell = +(us < ps),
    next_signal = un < 0.7 ? signal : -signal;
  return {
    ...outcome(h, q, signal, action, move, buy, sell, next_signal),
    p_buy: pb,
    p_sell: ps,
    uniforms: [...uniforms],
  };
}
export function transitions(h, q, signal, action, stress) {
  const rows = [],
    up = signal === 1 ? 0.65 : 0.35;
  for (const move of [-1, 1]) {
    const pm = move === 1 ? up : 1 - up,
      [pb, ps] = probabilities(q, action, move, stress);
    for (const buy of [0, 1])
      for (const sell of [0, 1])
        for (const ns of [-1, 1]) {
          const p =
            pm *
            (buy ? pb : 1 - pb) *
            (sell ? ps : 1 - ps) *
            (ns === signal ? 0.7 : 0.3);
          if (p > 0)
            rows.push({
              ...outcome(h, q, signal, action, move, buy, sell, ns),
              probability: p,
            });
        }
  }
  return rows;
}
export function train(episodes) {
  const rng = new Random(61237),
    table = Array.from({ length: 120 }, () => [0, 0, 0]),
    visits = Array.from({ length: 120 }, () => [0, 0, 0]),
    history = [];
  let running = 0,
    first_update = null;
  for (let episode = 0; episode < episodes; episode++) {
    let q = 0,
      signal = rng.draw() < 0.5 ? -1 : 1,
      total = 0;
    for (let h = HORIZON; h > 0; h--) {
      const state = index(h, q, signal),
        explore = rng.draw(),
        choice = rng.draw(),
        action = explore < 0.2 ? Math.floor(choice * 3) : greedy(table[state]),
        row = sample(
          h,
          q,
          signal,
          action,
          0,
          Array.from({ length: 4 }, () => rng.draw()),
        ),
        continuation =
          h > 1
            ? Math.max(...table[index(h - 1, row.next_q, row.next_signal)])
            : 0,
        target = row.reward + continuation;
      visits[state][action]++;
      const alpha = 1 / (1 + 0.05 * visits[state][action]),
        old = table[state][action];
      table[state][action] += alpha * (target - old);
      if (first_update === null)
        first_update = {
          state,
          action,
          old,
          target,
          alpha,
          new: table[state][action],
          transition: row,
        };
      total += row.reward;
      q = row.next_q;
      signal = row.next_signal;
    }
    running += total;
    if ((episode + 1) % 200 === 0) {
      history.push({
        episodes: episode + 1,
        mean_exploring_reward: running / 200,
      });
      running = 0;
    }
  }
  return {
    q_values: table,
    visits,
    policy: table.map(greedy),
    history,
    first_update,
    episodes,
    visited_entries: visits.flat().filter((v) => v > 0).length,
  };
}
export function solve(stress, policy = null) {
  const values = Array(120).fill(0),
    action_values = Array.from({ length: 120 }, () => [0, 0, 0]),
    selected = Array(120).fill(0);
  for (let h = 1; h <= HORIZON; h++)
    for (let q = -2; q <= 2; q++)
      for (const signal of [-1, 1]) {
        const state = index(h, q, signal);
        for (let action = 0; action < 3; action++) {
          let total = 0;
          for (const row of transitions(h, q, signal, action, stress)) {
            const future =
              h > 1 ? values[index(h - 1, row.next_q, row.next_signal)] : 0;
            total += row.probability * (row.reward + future);
          }
          action_values[state][action] = total;
        }
        selected[state] =
          policy === null ? greedy(action_values[state]) : policy[state];
        values[state] = action_values[state][selected[state]];
      }
  return {
    values,
    action_values,
    policy: selected,
    initial_value: (values[index(12, 0, -1)] + values[index(12, 0, 1)]) / 2,
  };
}
export function tape(count) {
  const rng = new Random(99881),
    rows = [];
  for (let i = 0; i < count; i++) {
    const signal = rng.draw() < 0.5 ? -1 : 1;
    rows.push({
      initial_signal: signal,
      draws: Array.from({ length: HORIZON }, () =>
        Array.from({ length: 4 }, () => rng.draw()),
      ),
    });
  }
  return rows;
}
const sum = (a) => a.reduce((a, v) => a + v, 0);
export function evaluate(policies, stress, paths) {
  const results = [];
  for (let p = 0; p < policies.length; p++) {
    const policy = policies[p],
      episodes = [],
      example = [];
    for (let ep = 0; ep < paths.length; ep++) {
      const path = paths[ep];
      let q = 0,
        signal = path.initial_signal,
        mid = 100,
        cash = 0,
        penalty = 0,
        fills = 0,
        inventory_squares = 0,
        total_reward = 0;
      for (let step = 0; step < path.draws.length; step++) {
        const h = HORIZON - step,
          action = policy[index(h, q, signal)],
          row = sample(h, q, signal, action, stress, path.draws[step]);
        cash +=
          row.sell * (mid + row.ask_distance) -
          row.buy * (mid - row.bid_distance) -
          row.fees;
        const next_mid = mid + row.move,
          marked = cash + row.next_q * next_mid;
        penalty += row.penalty;
        fills += row.buy + row.sell;
        inventory_squares += row.next_q ** 2;
        total_reward += row.reward;
        if (ep === 0)
          example.push({
            ...row,
            step: step + 1,
            mid,
            next_mid,
            cash,
            marked,
            cumulative_penalty: penalty,
            cumulative_reward: total_reward,
          });
        q = row.next_q;
        signal = row.next_signal;
        mid = next_mid;
      }
      const pnl = cash + q * mid - 0.25 * Math.abs(q);
      episodes.push({
        id: ep + 1,
        pnl_after_exit: pnl,
        inventory_penalty: penalty,
        objective: pnl - penalty,
        reward_sum: total_reward,
        reconciliation_gap: total_reward - (pnl - penalty),
        fills,
        inventory_squares,
        terminal_q_before_exit: q,
      });
    }
    const n = episodes.length;
    results.push({
      name: POLICIES[p],
      episodes,
      example,
      mean_objective: sum(episodes.map((r) => r.objective)) / n,
      mean_pnl: sum(episodes.map((r) => r.pnl_after_exit)) / n,
      mean_penalty: sum(episodes.map((r) => r.inventory_penalty)) / n,
      rms_inventory: Math.sqrt(
        sum(episodes.map((r) => r.inventory_squares)) / (n * HORIZON),
      ),
      mean_fills: sum(episodes.map((r) => r.fills)) / n,
    });
  }
  return results;
}
export function analyze(value = settings()) {
  validate(value);
  const c = { ...value },
    learned = train(EPISODES[c.budget]),
    baseline = Array(120).fill(2),
    oracle = solve(c.stress),
    learned_value = solve(c.stress, learned.policy),
    baseline_value = solve(c.stress, baseline),
    paths = tape(TEST_EPISODES),
    evaluations = evaluate(
      [learned.policy, baseline, oracle.policy],
      c.stress,
      paths,
    );
  [learned_value, baseline_value, oracle].forEach(
    (exact, i) =>
      (evaluations[i].exact_expected_objective = exact.initial_value),
  );
  const states = [];
  for (let h = 1; h <= 12; h++)
    for (let q = -2; q <= 2; q++)
      for (const signal of [-1, 1]) {
        const i = index(h, q, signal);
        states.push({
          id: i,
          remaining: h,
          inventory: q,
          signal,
          reachable_from_start: Math.abs(q) <= Math.min(2, 12 - h),
          learned_action: learned.policy[i],
          oracle_action: oracle.policy[i],
          q_values: learned.q_values[i],
          visits: learned.visits[i],
          oracle_value: oracle.values[i],
          learned_policy_value: learned_value.values[i],
        });
      }
  return {
    settings: c,
    training: learned,
    states,
    oracle,
    evaluations,
    exact_optimality_gap: oracle.initial_value - learned_value.initial_value,
    test_episodes: TEST_EPISODES,
    test_tape: paths,
  };
}
