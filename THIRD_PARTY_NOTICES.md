# Third-party source notices and attribution (working inventory)

This file indexes **known** third-party materials; it does not assert that every dependency, borrowed expression, included asset, or source lineage has been cleared.

| Component | Source | Known rights | Action |
| --- | --- | --- | --- |
| `src/cmbcst/mk_likelihood.py` | [Lonappan et al., `cosmic_birefringence`](https://github.com/antolonappan/cosmic_birefringence), `src/cosmic_bire/tools_fast.py` | MIT-licensed adapted/ported upstream expression | Preserve [complete upstream copyright and permission notice](THIRD_PARTY_LICENSES/cosmic_birefringence-MIT.txt); complete line-level provenance comparison |
| Underlying 2026 differential calibration work | [Lonappan, Keating & Arnold (2026)](https://arxiv.org/abs/2609.09149) | Published scientific work (not an original invention of this lab) | Cite accurately; distinguish reproduction from diagnostics |
| Planck PR4/NPIPE inputs | Planck project data and archives | Data provenance and use terms require source-specific review | Record archive product IDs, hashes, dates, citations and terms |
| Python runtime packages | `pyproject.toml` + resolved environment | Each independently licensed | Generate exact-environment SPDX SBOM and review transitive dependencies |
| External JoVin executable/harness | Locally configured external resource | **Not demonstrated as owned or licensed by this repository** | Record provider, source/license and execution scope privately |
| HTML/JS/CSS/graphics/fonts | `studio/`, `docs/` | Asset-level lineage unreviewed | Audit each embedded/imported third-party asset |

The upstream file itself records an antecedent code lineage from Johannes/J. R. Eskilt. Attribution and any further underlying license obligations must be checked rather than inferred from a single MIT notice.

A published article, shared equation, mathematical object, source dataset, and particular source-code expression are **not** interchangeable forms of copyright ownership. The project makes no exclusive claim to external scientific facts, data, or algorithms as abstract ideas.

**Pending:** A verified software bill of materials, license scan of all tracked code and assets, full Git-history source comparison, and independent legal review. No project-wide license is introduced here.
