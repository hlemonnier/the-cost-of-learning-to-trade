- NEVER write unit tests after you write code.
- Highly prefer E2E tests as the sole testing mechanism. Use them to verify complex features work. At the end of E2E tests, produce a verifiable and repeatable artifact.
- If you must test a system in isolation, FIRST write all the ways it could fail, THEN write the code.

# Shared research instructions

Read AGENTS.md and WORKLOG.md at the start of each session. Keep the worklog
compact: durable goal, accepted direction, verified evidence and next action.

BENCHMARK.md defines the synthetic market and core evaluation. The joint-belief
extension has a separate frozen protocol under extensions/bayes_adaptive/.

Preserve scientific equations, numerical conventions, seeds, data and negative
findings. Separate exact theoretical claims, numerical approximation and sampling
uncertainty. Synthetic returns do not establish real-market performance.

Use the trade_learning package and documented E2E checks. Retain JSON/CSV evidence.
Exclude temporary logs, environments and regenerable large control caches.

Maintain concise English documentation, working relative links and reproducible
commands. The public research edition uses the MIT licence.

## Cursor Cloud specific instructions

The default image has Python 3.12, but Ubuntu marks it externally managed and
does not ship `ensurepip`. `install` adds `python3.12-venv`, creates
`~/.local/share/trade-learning-venv`, installs `requirements-lock.txt`, and
performs an editable install. `/usr/local/bin/python`, `python3`, and `pip` are
wrappers for that environment. There is no `start` process and no secret.

Cold end-to-end acceptance matches GitHub Actions:

```bash
python verification/verify_pipeline.py --out outputs/reproduction/pipeline
```

It writes `verification.json` after two separate cold smoke runs (672 episode
rows, exact repeat). `outputs/reproduction/` is gitignored. The `full` profile
is a separate heavy solve and is not part of environment startup.
