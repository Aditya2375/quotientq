# QuotientQ

An original tabular Q-learning market-maker experiment. Train on sampled synthetic fills, freeze the table, then compare it with a fixed inventory-skew rule and an exact finite-model oracle on separate episodes. Inspect all 120 states, all three Q values and visit counts, all 512 paired test outcomes and the first episode's complete execution ledger.

**No market data, exchange connection, real orders, credentials or money. This is not a live bot, calibrated market model or evidence of expected real profit.** All prices are arbitrary units. The objective includes an invented inventory penalty, so it is not cash PnL.

## Run

Python 3.10+ standard library, no runtime packages:

```sh
python3 -m quotientq --budget 0 --stress 0 > experiment.json
python3 -m quotientq --budget 1 --stress 1 > stress-test.json
make test
```

```python
from quotientq import analyze, settings
config = settings()
config["stress"] = 1
result = analyze(config)
print(result["exact_optimality_gap"])
```

Browser interface, no build step:

```sh
python3 -m http.server 8000 --directory docs --bind 127.0.0.1
```

Open `http://127.0.0.1:8000`. Controls apply immediately; reset returns the default experiment. Selecting a policy-map cell changes only the audit, not the learned policy or test. Export contains the complete Q table, visit counts, training history, first TD update, oracle action values, all paired test episode summaries, first-episode event ledgers and raw test uniforms for replay. No download is sent anywhere.

## The finite market

There are 12 steps per episode. Mid starts at 100 and moves exactly ±1 after each interval's fills, so it stays positive. Cash and inventory start at zero; initial signal is a fair −1/+1 draw. The observed signal predicts the next up move with probability 0.35 when −1 and 0.65 when +1. Next signal equals the current one with probability 0.7, independent of the move and fill draws given the current signal.

State consists of **remaining steps 1..12, inventory −2..2, and signal −1/+1**: 120 table rows, three actions each. Time is included to make this finite-horizon problem Markov. Absolute mid and cash need not be state variables in this translation-invariant reward model, which assumes unlimited funding/borrow, fixed fees and no wealth constraint. This is not a general state sufficiency claim for markets.

| Action                | Bid distance below opening mid | Ask distance above opening mid |
| --------------------- | -----------------------------: | -----------------------------: |
| Tight                 |                            0.5 |                            0.5 |
| Wide                  |                            1.5 |                            1.5 |
| Inventory skew, long  |                            1.5 |                            0.5 |
| Inventory skew, short |                            0.5 |                            1.5 |
| Inventory skew, flat  |                            1.0 |                            1.0 |

All quotes are passive relative to the opening mid, not guaranteed passive relative to a later price. A buy is disabled at prior inventory +2 and a sell at −2. Both checks happen before either fill. An opposite-side fill does not reactivate a disabled side during that interval. Each enabled side fills at most one unit; both may fill. There is no queue priority, partial fill, latency, competition, spread process, cancellation race or external book.

Conditional on the **later** move, each enabled side independently fills with:

```text
p(side fill) = 0.4 / (1 + quoted distance) × multiplier
```

A buy on a down move or sell on an up move is adverse. Training and ordinary evaluation use adverse multiplier 1.5 and benign multiplier 0.5. Stress evaluation uses 2.5 and 0.2, without retraining. Maximum side probability is 2/3. The simulator samples the move before its conditional fills; this defines their joint law, not a future observation available to the policy. The policy sees only current state, never future move, uniforms or fills. These are Bernoulli toy fills, not Poisson intensities or exchange execution probabilities.

## Cash, reward and termination

Buy/sell fills execute at the opening quotes, paying fee 0.03 per filled lot. After those fills the mid moves. With `q` prior inventory and `q'` after fills:

```text
capture = buy × bidDistance + sell × askDistance
carry = q × move
fresh markout = (buy − sell) × move
marked-wealth change = capture + carry + fresh markout − makerFees
inventory penalty = 0.08 × q'²
step reward = marked-wealth change − inventory penalty − terminal exit cost
```

At the true terminal step, force exit at final mid minus 0.25 for a long or plus 0.25 for a short. Cost is `0.25 × abs(q')`; there is no additional exit fee. Unlimited exit liquidity is assumed. The summary's terminal inventory is before this forced exit; afterwards the conceptual position is flat. First-episode cash and marked columns are before exit, while its last reward already deducts exit cost.

```text
PnL after exit = cash + terminalInventory × finalMid − exitCost
objective = PnL after exit − sum(inventory penalties)
```

The sum of step rewards reconciles to that objective. Exported reconciliation gaps expose floating-point error, not forced zeros. Capture and markout are accounting components, not separate realized profit. RMS inventory averages squared after-fill inventory over all test steps; fills count both sides. No annualization, risk-adjusted performance, half-life, capital return or solvency claim is made.

## Actual sampled Q-learning

Zero-initialize 360 Q entries. Train for 2,000 or 12,000 episodes on the ordinary fill law only. At each step, ε-greedy chooses a uniform random action with probability 0.2 and otherwise a greedy action. Exploration's uniform action draw is consumed even when not used. The undiscounted update is:

```text
N(s,a) += 1
alpha = 1 / (1 + 0.05 × N(s,a))
target = reward + max_a' Q(nextState,a')   # nonterminal
       = reward                          # true terminal step
Q(s,a) += alpha × (target − Q(s,a))
```

The terminal exit cost is already in reward. There is no continuation after step 12. These are sample transitions and max-next-Q targets, not exact model-based training, SARSA, a hand-coded quoting formula or a neural network. No oracle value, transition enumeration or test episode enters training. `train` can run with the model solver unavailable.

For greedy numerical ties, choose the first action within 1e−12 of the actual maximum, in order tight, wide, skew. Raw Q values remain unchanged. The same tie policy applies to the oracle; its result is optimal up to that numerical tolerance rather than an exact-arithmetic proof. Every policy is frozen before evaluation. Finite budgets and a fixed seed establish **no convergence or robustness guarantee**. Increasing the budget need not monotonically improve Q values, policy or outcomes.

All states are exposed, including 12 impossible early-horizon states. From zero inventory, a state is reachable only if `abs(q) <= min(2, 12 − remaining)`. Thus 108 states / 324 state-action entries are potentially reachable. The displayed denominator 360 includes the unreachable entries; untouched Q values are zero, not confident estimates. Small budgets may also miss reachable entries. Policy cells mark unreachable states and disagreements with the test oracle.

Training-history points are mean **exploring** reward in blocks of 200 episodes, not greedy test returns. The larger budget contains the entire shorter training prefix. No checkpoint selection, early stopping, learned seed choice or hyperparameter search is performed. Manually choosing a budget after inspecting the test is not an unbiased external evaluation.

## Frozen test and exact benchmark

The test is 512 episodes with a separate fixed PRNG seed and no exploration. Learned, fixed-skew and model-oracle policies share the base uniforms and initial signals. Their inventory, quotes, probabilities and actual fills can differ; they are not competing for the same observed orders. The fixed benchmark always selects action 2 and still honors caps.

The oracle separately enumerates every positive-probability combination of move, buy, sell and next signal for each state/action and solves the finite MDP by backward induction. Fixed-policy backward induction evaluates the learned and fixed-skew policies without changing them. The test's initial expected objective averages signal −1 and signal +1 at remaining 12, inventory 0. The reported gap is oracle expected objective minus learned-policy expected objective under the selected toy test law.

This expected objective bound applies only within the declared action set, known transitions, reward and horizon, up to numerical tolerance. It is **not** a pathwise cash bound, real-market bound, causal future-price predictor or empirical profitability result. In stress mode the oracle knows the changed fill law and is explicitly not a fair equally uninformed learner. Finite sampled policy means can violate an expected-value ordering without violating the model calculation. The paired difference histogram is descriptive, not a confidence interval, p-value, market-alpha result or test of significance. No multiple-training-seed analysis is claimed.

## Determinism and boundaries

The original 32-bit LCG is `(1664525*s + 1013904223) mod 2^32`, with uniform `(s + 0.5)/2^32` in (0, 1). Training seed 61237, test seed 99881. Each training episode consumes an initial-signal draw, then six uniforms per step: explore, action, move, buy, sell, next signal. Each test episode consumes initial signal then four per step: move, buy, sell, next signal. Both budgets have the same prefix. These are invented pseudo-random samples, not cryptographic randomness or independent market datasets.

`analyze(config)` accepts exactly `budget` and `stress`, built-in integers 0 or 1. Python requires a built-in dict and rejects booleans/floats/extra or missing keys; omitted config or `None` gives defaults. JavaScript accepts a plain object with the same integer domain; omitted/undefined gives defaults and null is invalid. Settings and results are detached between calls. Internal helpers assume this finite trusted domain, not arbitrary arrays, uploaded models, proxies, custom subclasses or untrusted data. No serialized model is loaded.

## Verify

```sh
npm ci --ignore-scripts
npx playwright install chromium
npm test
npm run test:browser
```

Runtime has no dependencies. Development-only Playwright 1.56.1 is locked. Native tests independently reconstruct both complete training runs with a second integer generator and algebraically different TD update, enumerate every transition law and Bellman action value, replay every test episode for all policies, reconcile cash/reward and terminal cost, check reachable states and no oracle leakage, and verify strict CLI inputs. Python/JavaScript tests compare all fields of all four complete states, including the full test tape, tables, oracle values, episode results and first-episode ledgers. Browser tests inspect all displayed cells, all policy-map actions/reachability/disagreements, selected state audits, reset, export, invalid-input retention and privacy at five widths. Shared bugs can survive parity; finite tests do not validate real markets.

CI uses Python 3.10/3.12/3.13, Node 22, SHA-pinned Actions, verified Gitleaks and pinned Semgrep with explicit DOM/eval/runtime-network/native-effect rules. The app has no runtime network requests, telemetry, cookies, local storage, service worker or user uploads. It is a static five-file site with original SVG/styles and no external fonts/assets. Meta CSP denies connections, objects, base URLs and forms, with same-origin scripts/styles/images only. Host request logs/headers/framing/caching and browser history/download/session restore remain outside app control; reset is not secure erasure. This is not a penetration-test or security certification.

Native controls/details, labelled canvases, full numerical tables, a skip link, visible keyboard focus and reduced motion are included. Policy maps are labelled native buttons. Wide tables scroll locally; selection is indicated by a border and aria-pressed, not color alone. Canvases are noninteractive and have nearby data/export alternatives. No assistive-technology or cross-browser certification is claimed.

## Concept sources and originality

Original custom finite MDP, implementation, simulated data, styles and SVG. No source code, model weights, dataset, empirical results or assets copied from references. In particular, this is not a reproduction of a cited empirical market-making strategy.

- [Watkins & Dayan, 1992, author-hosted primary paper](https://www.gatsby.ucl.ac.uk/%7Edayan/papers/cjch.pdf): tabular Q-learning and limiting convergence assumptions, not a finite-budget guarantee.
- [Spooner et al., 2018, Market Making via Reinforcement Learning](https://arxiv.org/pdf/1804.04216): inventory-risk and reward-design motivation only.
- [Farama, termination versus truncation](https://farama.org/Gymnasium-Terminated-Truncated-Step-API): finite-horizon time in state and zero terminal continuation.
- [Gymnasium time-limit reference](https://gymnasium.farama.org/v1.3.0/tutorials/gymnasium_basics/handling_time_limits/): true termination treatment.
- [Dive into Deep Learning, Q-learning](https://www.d2l.ai/chapter_reinforcement-learning/qlearning.html): ε-greedy and off-policy concepts. Example code is not used.
- [Stable Baselines3, evaluation tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html): separate deterministic evaluation; no SB3 runtime used.
- [Peitz, Q-learning course](https://sebastianpeitz.github.io/DeepRL_Course/08-q-learning-deck.html): tabular max-next-Q versus SARSA.

MIT © 2026 Aditya Kulkarni.
