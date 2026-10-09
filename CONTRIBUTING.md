# Contributing

State the question, assumptions, failure modes and acceptance criteria before
changing a model or numerical method. Preserve the frozen protocol and adverse
findings; use a separate study and fresh streams for new hypotheses.

Use E2E experiments for behaviour changes:

```bash
python verification/verify_pipeline.py --out outputs/reproduction/YOUR_RUN
```

Retain machine-readable evidence. Do not add unit tests after the code they test.
Keep PnL distinct from the penalised objective, theoretical bounds from numerical
approximations, and sampling uncertainty from numerical error.

Use concise English documentation, update WORKLOG.md after meaningful changes,
and exclude temporary logs, environments and regenerable control caches.
