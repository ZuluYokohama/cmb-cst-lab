# Provenance evidence boundaries

This directory is a **public index, not the confidential evidence archive**. It records conservative classifications needed to tell apart reproduced science, AI-assisted project code, external implementations, generated outputs, and unresolved lineage.

- `component_registry.json` declares a limited set of **evidence-supported classifications**. Any tracked file not declared defaults to `UNRESOLVED`. Classifications are not legal title conclusions.
- Run `python scripts/provenance_audit.py check` for lightweight non-destructive integrity and notice checks.
- Run `python scripts/provenance_audit.py inventory` for a JSON report of tracked paths, SHA-256 hashes, declared status and unresolved counts. This report contains repository paths but **does not attest originality**.
- Run `python scripts/provenance_audit.py snapshot --out ../cmb-cst-private-snapshot` from a trusted local working copy for a Git bundle, refs, commits and hash-verified audit manifest. The output directory must be **outside the public repository**. Protect it as confidential.
- Preserve actual dated human design notes, AI session exports, human edits, rejected proposals, upstream comparisons, contracts and verification logs in a separate access-restricted archive. Never upload raw personal chats, passwords, secrets, or confidential materials as public evidence.

A SHA-256 hash permits later integrity comparison; by itself it proves neither who created the content nor when it existed. The tool does not audit AI-provider contracts, copyrightability, patents, third-party assets, Git history secrets or scientific validity.

## Claim-to-evidence chain

`requirement → source/antecedent → implementation commit → test/fixture → result + inputs → independent comparator → limitations → rights/disclosure gate`

For every **novel** claim, keep the antecedents and a falsification test. For every **copyright** claim, separately document *eligible human-authored expression* and chain of title. For every **patent** claim, obtain counsel's review before new public enabling disclosure.

See [value-to-evidence verification plan](../docs/VALUE_GENESIS_VV_PLAN.md), [third-party notices](../THIRD_PARTY_NOTICES.md), and [rights and reuse](../RIGHTS_AND_REUSE.md).
