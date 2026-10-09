# Current-delivery failure contract — PER-505, second review

Declared before changing the reporting pipeline. The original successful receipts
remain identifiable at `audit/reviewed-1174293` and will be archived with that
revision rather than silently reinterpreted as current evidence.

## Failure scenarios

1. A successful report receipt contains the original primary estimate or stress
   result while the current summary/claims contain the corrected result.
2. A receipt binds a path but not its bytes, so source, raw data, summary, generated
   claims, TeX inputs, figures or PDFs change without invalidating it.
3. A compile receipt exists but refers to an old PDF, old generated inputs, a failed
   compiler invocation or only a partially completed two-document build.
4. Running report generation directly refreshes the text but leaves an old
   unqualified success receipt. A stdout redirect is not a durable status API.
5. Compilation or validation fails yet a previously successful current receipt
   survives as if it described that attempted build.
6. The main PDF exceeds eight pages, omits required claims, or lacks current
   exact-byte visual inspection; a successful compiler exit does not close these.
7. An archive contains receipts whose source/data/claim/PDF identities disagree,
   misses mandatory current receipts, or includes stale unlabelled copies.
8. A verifier trusts one `passed` flag without checking its input identities and
   result arithmetic against the authoritative summary/claims.
9. Repeated runs serialize different self-referential hashes or include the archive
   in its own input inventory. Hash inputs must be acyclic and explicit.
10. A current completion narrative says regeneration is pending despite finished
    evidence, or relabels local validation as a new external-reviewer sign-off.
11. Expanded review evidence pushes a single Git-tracked archive beyond GitHub's
    file-size limit. If a companion archive is needed, no evidence may be omitted:
    one inventory/status must authenticate both archives, every member must match
    the current tree, and a missing companion must make full-bundle validation fail.

## End-to-end acceptance

Run report generation, compile both PDFs, validate current receipts against actual
files and canonical results, render every page for primary-agent review, and package
the exact inspected bytes. Exercise the command-line validator on a temporary copy
of the delivered file tree with deliberate stale primary/source/claim/PDF/build
receipts and require rejection. Save the baseline identities, mutation results and
repeatable commands as an artifact. No production simulation changes are needed.
