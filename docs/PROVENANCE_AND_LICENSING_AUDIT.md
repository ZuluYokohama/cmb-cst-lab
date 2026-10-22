# Provenance and licensing audit (working record)

Date: 2026-10-22. This is a factual audit note, not a declaration of exclusive ownership or a project-wide license.

## Current permissions

- The public repository does not currently contain a top-level project LICENSE. Visibility and GitHub's platform permissions are not a general open-source grant.
- Do not infer that the user, an AI agent, or the repository owns the mathematical method, the public Planck data, or code developed by others.
- Do not claim all generated code is exclusively copyrightable. Human-authored source, creative human edits, and original creative selection/arrangement may be eligible depending on the specific facts.
- **No new license covering original project files is granted by this document.** Selecting MIT, Apache, AGPL, proprietary terms, or dual licensing requires an explicit rights and reuse decision.

## Identified upstream component

File: `src/cmbcst/mk_likelihood.py`.

Its module header describes it as a NumPy port of the model in:
- Anto Idicherian Lonappan et al., upstream repository https://github.com/antolonappan/cosmic_birefringence
- Source: `src/cosmic_bire/tools_fast.py`
- Upstream license: MIT; original copyright line identifies **2026 Anto Idicherian Lonappan**.
- Upstream also notes an earlier code lineage ('from Johannes code'); review that provenance before final redistribution conclusions.
- Original license: https://github.com/antolonappan/cosmic_birefringence/blob/main/LICENSE

**Current action completed on this draft branch:** the upstream copyright and complete MIT permission notice are retained verbatim in `THIRD_PARTY_LICENSES/cosmic_birefringence-MIT.txt` (checked against the upstream file on 2026-10-22). Attribution alone would not be a substitute. The precise source correspondence and earlier code lineage still need independent review before a redistribution or legal clearance conclusion.

## Additional materials requiring separate rights analysis

- Planck PR4/NPIPE input data: scientific source and credit requirements, not owned by this repository.
- Python package dependencies: independently licensed; see `pyproject.toml`.
- `scripts/jovin_verify.py`: a local wrapper calling a configured external JoVin harness. Presence of this wrapper does **not** establish ownership of the external harness executable or its license.
- `studio/` and `docs/`: the live studio and generated page reference Google Fonts externally (Spectral, IBM Plex Sans, IBM Plex Mono), and contain scientific source links. These are not locally bundled font binaries; attribution, privacy/hosting and asset permissions still need source-specific review.
- Manuscript, figures and simulation outputs: establish actual authorship, upstream contributions, citations and AI assistance separately.

## AI-assisted development evidence to preserve

For each substantial file/module, retain:
1. Source of requirements and architectural direction, including dated human-authored decisions.
2. AI tool/model sessions used to generate or revise it; note what remains unavailable.
3. Third-party sources consulted, copied, translated, or adapted and their license conditions.
4. Specific human-authored revisions, selection/arrangement and verification actions.
5. Test receipts, exact input hashes, commit hashes, and reproducibility instructions.

Git commits, prompts, and tests are valuable provenance, but **do not alone prove legally sufficient human authorship**.

## Public configuration and exposure review

- `.mcp.json` contains an absolute local Windows executable path and a named WSL runtime operating as root. No credential-like literal was observed in this reviewed file.
- `scripts/jovin_verify.py` also embeds local runtime paths. Review whether these implementation details should remain public. Removing them in future commits cannot erase existing Git history.
- GitHub Actions currently checks Python tests on Linux and Windows. Its pass status validates only the executed tests and assumptions, not scientific novelty or third-party licensing.
- A limited static pattern scan of selected tracked files found no obvious API tokens or private-key blocks. This is **not** full secret scanning of every file, branch, or history.

## Outstanding checks

- Determine if AI-generated code reproduces protected expression from sources other than the cited MIT repository.
- Check source history and any copied dependencies/assets, generated HTML links, build scripts, embedded images and exact code lineage.
- Review GitHub secret-scanning alerts and history with appropriate access.
- Validate licensing strategy for desired scientific reproducibility and potential enterprise reuse.
- Seek qualified IP counsel if formal ownership, commercial exclusivity, a trademark filing, or technology transfer is planned.

Documentation: U.S. Copyright Office AI Part 2 (Jan. 2025), https://www.copyright.gov/ai/ ; GitHub licensing https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository ; MIT conditions https://opensource.org/license/mit .
