# Literature Review Plan: Fair Algorithmic Recourse with Kolmogorov-Arnold Networks

**Author:** Tiago Silva · MSc in Informatics Engineering, specialization in Artificial Intelligence (DEI, University of Coimbra)
**Supervisors:** Prof. Pedro Henriques Abreu (DEI, University of Coimbra) · Prof. Bruno Veloso (INESC TEC; Faculty of Economics, University of Porto)
**Intermediate delivery:** January 2027. The plan assumes **~22 Jan 2027**; confirm the official date and shift the last two weeks if needed.
**Plan start:** Tue 27 Oct 2026, which gives about 13 weeks including the Christmas break.
**Slides for the supervisors:** `presentation/lit_review_plan/presentation.tex` (same grouping as this plan).
**Extraction spreadsheet:** `D:\Literature Review.xlsx` (28 papers logged as of 29 Oct 2026).

---

## 0. What the January delivery must contain

I based this on the three DEI reports supervised by Prof. Abreu that were provided as examples: Mercier 2018 (intermediate report), Cortesão 2024 and Cruz Jan-2026 (the closest model: same supervisor, same specialization, same delivery point).

| Template file | Chapter | Content | Target pages |
|---|---|---|---|
| `text/1-introduction.tex` | 1 Introduction | Context, objectives/RQs, **work plan (Gantt, 1st sem. actual vs planned + 2nd sem.)**, **risk analysis table**, document structure | 5–7 |
| `text/2-background.tex` | 2 Background Knowledge | Self-contained theory: notation, KANs, group and individual fairness, causality, counterfactuals/recourse | 15–18 |
| `text/3-literature-review.tex` | 3 Literature Review | **The core of the delivery.** 3.1 review method · 3.2 G1 counterfactual search · 3.3 G2 plausibility and causality · 3.4 G3 fairness of recourse · 3.5 G4 KAN components · 3.6 synthesis and research gaps. Comparison tables and a "summary and implications" per section | 25–32 |
| `text/4-proposed-approach.tex` | 4 Proposed Approach | Problem formulation, candidate architectures (decision matrix), experimental design (datasets table, metrics, baselines) | 5–8 |
| `text/5-conclusion.tex` | 5 Conclusion | First-semester summary, next steps | 1–2 |

Overall target: **~55–70 pages of main text and 100–150 references.** Cruz's report has about 65 pages of text plus 24 pages of references.

Frontmatter to fill in during the final weeks (template files, format unchanged): abstract EN/PT, keywords, acronyms (`frontmatter/8-acronyms.tex`), acknowledgements and the **AI-use declaration** (`frontmatter/5-ai-declaration.tex`). Record AI-tool use as you go so the declaration is accurate.

---

## 1. Scope: one group per part of the recourse pipeline

```
                 G1 Search            G2 Plausibility and causality
                     │                         │
  x, h(x)=0 ──► [Recourse generator] ──a──► [Causal model] ──x^SCF──► [Decision model h] ──► h(x^SCF)=1
                     │
                     └──► G3 Fairness of recourse: cost c(a*; x) compared across protected groups

  G4 KAN components: any of the three boxes can be a KAN
  Not tied to one part: Foundations (surveys, position papers) · Evaluation (datasets, metrics, tools)
```

| Group | Question it answers | Logged (29 Oct) | Ch. 3 |
|---|---|---|---|
| **G1 Counterfactual search** | How is a valid counterfactual found, and which model access does it need? | 7 | §3.2 |
| **G2 Plausibility and causality** | Does the counterfactual stay on the data and respect causal structure? | 9 | §3.3 |
| **G3 Fairness of recourse** | Is the cost of recourse equal across groups and individuals? | 2 (+1 shared with G1, +5 seeds not yet logged) | §3.4 |
| **G4 KAN components** | Which components of the pipeline have been built with KANs, and with what evidence? | 4 | §3.5 |
| Foundations | Surveys and position papers | 5 | Ch. 2, §3 intro |
| Evaluation | Datasets, metrics, tools | 1 | every section's table, Ch. 4 |

**Why this grouping replaced the three topic clusters** (CE and fair recourse / causal modelling / KANs), based on the spreadsheet on 28 Oct 2026:
- Tab A held 17 of 27 papers: surveys, search methods, plausibility methods, a fairness paper, a KAN paper and a tool.
- Plausibility methods sat in two tabs: FACE, REVISE, prototypes and DACE in A; CRUDS and Mahajan et al. in B.
- Fairness of recourse had no tab: 2 papers logged, 6 more only in `references.bib`.
- Tab C mixed 2 surveys, 1 out-of-scope application, a causal model and a fairness model; the KAN-for-CE paper sat in A.
- The datasets and metrics tab was empty.

Mapping of the 27 logged papers: G1 ← 7 from A; G2 ← 4 from A + 5 from B; G3 ← 1 from A; G4 ← 1 from A + 3 from C; Foundations ← 3 from A + 2 from C; Evaluation ← 1 from A.

**Intersections found so far (spreadsheet, 29 Oct 2026; to be confirmed by the systematic search):**

| Intersection | Logged papers | Count |
|---|---|---|
| Recourse × causal structure | Lash et al. 2018; Mahajan et al. 2020; Karimi et al. 2020, 2021; von Kügelgen et al. 2022 | 5 |
| Recourse × fairness | Sharma et al. 2020; von Kügelgen et al. 2022; Ezzeddine et al. 2026 | 3 |
| CE × KANs | Valencia et al. 2026 (RL agents, not tabular classification) | 1 |
| Causal model × KANs | Almodóvar et al. 2026 (KaCGM) | 1 |
| Fairness × KANs | Priyadarshini & Gago-Masague 2025 (fair representations) | 1 |
| **Fair recourse × KANs** | — | **0** |

Also relevant: a KAN imbalanced-data study (Yadav & Vijay, 2025) states that KAN bias mitigation on tabular data is unvalidated, and *Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden* (Barrainkua et al., AAAI-26) is the latest fair-recourse seed.

### The review must decide the architecture
You are not yet sure of the architecture, so Chapter 3 should be written to **end in a decision**. Each candidate role for the KAN raises its own questions, and each is informed mainly by one or two groups:

| Option | Role of the KAN | Main groups | What the review must answer |
|---|---|---|---|
| **O1. KAN as the decision model** | Classifier whose additive univariate structure makes recourse cheaper to compute and easier to explain. A fairness-of-recourse term is added during training. | G1, G3, G4 | Which search methods work with a KAN (gradients, solver encodings)? Which in-training recourse-fairness regularizers exist? Are KANs competitive on tabular data? |
| **O2. KAN as the causal model** | KANs parameterize the structural equations of an SCM used to compute causally consistent recourse (interventions, not just feature changes). | G2, G4 | Which causal generative models exist, and what are their identifiability guarantees and data types? What does KaCGM already show? Where does the causal graph come from? |
| **O3. KAN as the recourse generator** | Amortized model that maps an individual to a recommended action, with fairness constraints. | G1, G3 | Which amortized/generative recourse methods exist? How do they handle actionability, plausibility and fairness constraints? |
| **O4. Combination** | Any of O1–O3 together | all | Trade-offs in complexity and evaluability |

Whichever option is chosen, **fairness** enters as a penalty or constraint on the cost of recourse across protected groups (G3). Chapter 4 then justifies the choice with this table, filled in from the evidence.

---

## 2. Preliminary research questions (refine with the supervisors in W1, freeze in W8)

| RQ | Question | Groups | Answered by |
|---|---|---|---|
| **RQ1** | How can KANs be integrated into a recourse pipeline (as decision model, causal model or recourse generator) so that recommendations are valid, actionable and causally consistent? | G1, G2, G4 | Review → architecture decision (Ch. 4) |
| **RQ2** | Which fairness-of-recourse notions are meaningful for KAN-based models (group cost parity, social burden, causal fairness of recourse), and how can they be measured? | G3 (G2 for causal notions) | Review → fairness-of-recourse notions table |
| **RQ3** | Does the interpretability of KANs (edge functions, pruning, symbolic formulas) help to *audit and explain* recourse disparities better than MLP-based models? | G3, G4 | Experiments (2nd semester) |
| **RQ4** | What is the trade-off between predictive performance, recourse cost and recourse fairness? | G3, G4 | Experiments (2nd semester) |

---

## 3. Search protocol (semi-systematic)

None of the example reports used a formal protocol. A documented protocol is still cheap to add (1–2 pages, §3.1). It makes your gap claims defensible ("no study in our search combined X and Y"). Follow Kitchenham & Charters (2007) for the protocol, PRISMA 2020 for the flow diagram and Wohlin (2014) for snowballing.

### 3.1 Sources
- **Databases:** Scopus, Web of Science, ACM Digital Library, IEEE Xplore, DBLP.
- **Preprints (mandatory here):** arXiv, Semantic Scholar, Google Scholar. The KAN literature is from 2024 onwards and is mostly preprints.
- **Venues to scan manually (2019–2026):** FAccT, AIES, NeurIPS, ICML, ICLR, AAAI, IJCAI, AISTATS, UAI, CLeaR, KDD, TMLR, JMLR. Journals: ACM Computing Surveys, Data Mining and Knowledge Discovery, Artificial Intelligence, Machine Learning, IEEE TNNLS, IEEE TAI.

### 3.2 Search strings (title/abstract/keywords; adapt syntax per database)

| ID | Feeds | String |
|---|---|---|
| S1 | G1, G2 | `("algorithmic recourse" OR "actionable recourse" OR "counterfactual explanation*" OR "counterfactual recourse")` |
| S2 | G3 | `S1 AND (fair* OR bias OR discriminat* OR "equal* recourse" OR "recourse disparit*" OR "social burden")` |
| S3 | G2 | `("causal recourse" OR ("recourse" AND "structural causal model*") OR ("counterfactual inference" AND ("normalizing flow*" OR variational OR diffusion OR "neural causal model*")))` |
| S4 | G4 | `("Kolmogorov-Arnold network*" OR "Kolmogorov Arnold network*") AND (interpretab* OR explainab* OR "symbolic regression" OR fair* OR counterfactual OR causal OR recourse OR tabular)` |

S1 hits are assigned to G1 or G2 during screening by their main contribution (how the counterfactual is searched vs. what constrains where it lands). The KAN term on its own (S4 without the `AND` clause) returns thousands of application papers. Screen **S4** fully, and use the bare KAN term only to find surveys and the most-cited methodological variants.

### 3.3 Inclusion and exclusion criteria
- **IC1** Proposes or evaluates a method, metric, benchmark or theoretical result on counterfactual search, plausibility or causal consistency of recourse, fairness of recourse, causal counterfactual inference, or KAN methodology.
- **IC2** Applicable to tabular data.
- **IC3** Peer-reviewed. A preprint qualifies only if it is highly cited, or directly on-topic and less than 12 months old.
- **IC4** Written in English.
- **Time windows:** G1 and G2 2017–2026 (from Wachter et al.), plus the foundational causality books; G3 from 2019 on; G4 2024–2026, plus the classical theorem papers. Foundational general-fairness papers (2012–2017) are cited in Ch. 2 only.
- **EC1** Image/text-only CE with no transfer to tabular data.
- **EC2** "Recourse" in the legal or operations-research sense.
- **EC3** KAN application papers with no methodological contribution (count them, do not review them).
- **EC4** Abstracts, posters or theses.
- **EC5** Full text unavailable.
- **Open scope decision (W1 meeting):** whether counterfactual work on RL agents (e.g. Valencia et al. 2026) is in scope or excluded by IC2.

### 3.4 Screening pipeline (keep a log of counts for the PRISMA diagram!)
1. Run S1–S4 in every source, export to Zotero, deduplicate, and **record counts per source**.
2. Screen by title/abstract (fast: include / exclude / maybe).
3. Screen the full text of the includes and maybes, and fill the extraction spreadsheet (§4).
4. Snowball: one backward and one forward round from the seed set (`references.bib`) and the strongest includes.
5. **Freeze the systematic search around 20 Dec 2026.** Papers found later enter only as "recent work" if crucial.

---

## 4. Tooling and note-taking workflow

- **Zotero + Better BibTeX** auto-exports `references.bib` with citekeys in the same style as the seed file (`karimi2021algorithmic`).
  - One collection per group (G1–G4), plus `Foundations`, `Evaluation`, a `Background` collection for Ch. 2-only sources and a `Methodology` collection.
  - Tags: `screen:in`, `screen:maybe`, `read:skim`, `read:deep`, `rel:1/2/3`.
- **Extraction spreadsheet** (`D:\Literature Review.xlsx`): one tab per group (G1, G2, G3, G4, Foundations, Evaluation), one row per included paper. Current columns: title, authors & year, what it talks about, full citation, limitations/future work. Add:
  `citekey | year | venue | model class (linear/tree/MLP/KAN) | model access | causal input (none / constraints / graph / full SCM) | fairness notion (group / individual / causal / burden) | recourse cost definition | actionability handling | datasets | metrics | code? | relevance (1–3)`
- **Five-line note per deep-read paper:** problem, method, assumptions, key result, limitation/relevance to me. These notes become paragraphs almost directly.
- **Snowballing:** Semantic Scholar "cited by", ResearchRabbit or Connected Papers.
- **Alerts:** Google Scholar alerts for `"Kolmogorov-Arnold" recourse`, `"Kolmogorov-Arnold" fairness` and `"algorithmic recourse" fairness`, plus arXiv cs.LG keyword alerts. Spend **30 minutes every Monday** triaging them.
- **Write-as-you-read rule:** each group week ends with that section drafted in LaTeX, even if rough.

---

## 5. Groups, seed papers and must-answer questions

`references.bib` mirrors the spreadsheet: it holds exactly the 28 logged papers. **Keys marked ○ are planned seeds that are not logged yet**, so they are not in `references.bib`: add each one to the spreadsheet and to `references.bib` together when you read it (the earlier 66-entry seed file is in git commit eb06687). **Verify the volume, pages and venue of every entry against DBLP or the publisher before final submission.**

### G1. Counterfactual search (§2.5, §3.2)
- **Seeds:**
  - Exact solvers: `ustun2019actionable` (IP), `russell2019efficient` (MIP), `karimi2020model` (MACE, SMT)
  - Gradient-based: `wachter2018counterfactual`○, `mothilal2020explaining` (DiCE)
  - Evolutionary and surrogate: `dandl2020multiobjective` (MOC), `sharma2020certifai` (CERTIFAI, also G3), `white2019measurable` (CLEAR)
  - Robust and global: `upadhyay2021towards`○, `rawal2020beyond`○
- **Must answer:** Which search strategies exist, and which model access does each need? Which apply to a KAN directly (gradients) and which would need a solver encoding of spline functions? Which methods are amortized (relevant to O3)? How do robust and global methods change the search objective?
- **Deliverable:** **search-methods table** (method | year | search strategy | model access | diversity? | robustness? | datasets | metrics | code).

### G2. Plausibility and causality (§2.4, §3.3)
- **Seeds:**
  - Data manifold: `joshi2019towards` (REVISE), `pawelczyk2020learning`○ (C-CHVAE), `poyiadzi2020face` (FACE), `vanlooveren2020interpretable` (prototypes), `kanamori2020dace` (DACE), `downs2020cruds` (CRUDS)
  - Causal constraints and interventions: `lash2018prophit` (Prophit), `mahajan2019preserving`, `karimi2021algorithmic`
  - Imperfect causal knowledge: `karimi2020algorithmic`, `dominguezolmedo2022adversarial`○
  - Deep causal generative models: `pawlowski2020deep`○, `khemakhem2021causal`○, `sanchezmartin2022vaca`○, `javaloy2023causal`○, `chao2023interventional`○, `xia2023neural`○
  - Foundations (Ch. 2): `pearl2009causality`○, `peters2017elements`○
- **Must answer:** What separates data-manifold plausibility from causal consistency, and when is each enough? What is the difference between counterfactual explanations and causal recourse? What happens when the SCM is unknown or misspecified? Which deep causal models support counterfactual (rung-3) queries, under which assumptions, and on mixed-type data? How is counterfactual quality evaluated? Which have interpretable mechanisms (relevant to O2)?
- **Deliverable:** **plausibility and causal models table** (method | year | mechanism | causal input | mixed data? | interpretable mechanisms? | datasets | metrics).

### G3. Fairness of recourse (§2.3, §3.4)
- **Seeds:**
  - Logged: `sharma2020certifai` (burden), `vonkugelgen2022fairness` (group and individual fairness of causal recourse; cite the AAAI 2022 version, the spreadsheet has the arXiv preprint), `ezzeddine2026fair` (individual and group fairness in generation)
  - Not yet logged: `gupta2019equalizing`○, `ehyaei2023robustness`○, `kavouras2023fairness`○, `bell2024fairness`○, `barrainkua2026revisiting`○
- **Must answer:** Which recourse-fairness notions exist (distance gap, burden, cost parity, individual/causal fairness of recourse, substantive equality, worst-case social burden), what does each compare, and which require an SCM? Are there in-training or generation-time mitigation methods (O1, O3) or only post-hoc audits? How do fair classifiers interact with fair recourse (the Barrainkua et al. finding)?
- **Deliverable:** **fairness-of-recourse notions table** (notion | definition | compares | needs SCM? | mitigation? | datasets).

### G4. KAN components (§2.2, §3.5)
- **Seeds:**
  - Core model (Ch. 2): `kolmogorov1957representation`○, `arnold1957functions`○, `girosi1989representation`○, `liu2025kan`○, `liu2024kan`○
  - Variants and tabular evidence: `yu2024kan`○, `li2024kolmogorov`○, `poeta2024benchmarking`○, `yadav2025kolmogorov`○
  - KANs in the recourse pipeline: `almodovar2026kolmogorov` (KaCGM, causal generative model), `valencia2026counterfactual` (CEs for an RL agent), `priyadarshini2026learning` (fair representations)
  - Out of scope, counted only: `hoang2026ultrafast` (on-chip online learning)
  - Related interpretable additive models: `agarwal2021neural`○, `rudin2019stop`○
- **Must answer:** Which variants exist (B-spline, RBF, Chebyshev, wavelet, Fourier, ...) and what are their trade-offs? Do KANs actually beat MLPs on tabular data under fair comparison? What evidence supports their interpretability claims? Which pipeline components have already been built with KANs, and what do KaCGM and Valencia et al. leave open? **Confirm the gap** (fair recourse × KANs).
- **Deliverable:** **KAN uses table** (work | year | KAN role in the pipeline | data | interpretability evidence | relation to the thesis), and the explicit gap statement.

### Foundations (§3 intro, Ch. 2)
- Recourse and CE surveys and positions: `karimi2022survey`, `verma2024counterfactual`, `guidotti2024counterfactual`○, `venkatasubramanian2020philosophical`, `barocas2020hidden`○
- KAN surveys: `somvanshi2025survey` (ACM Computing Surveys, 2025), `hou2024comprehensive`○, `noorizadegan2025practitioners`
- General fairness (§2.3): `mehrabi2021survey`○, `caton2024fairness`○, `barocas2023fairness`○, `dwork2012fairness`○, `hardt2016equality`○, `kleinberg2017inherent`○, `chouldechova2017fair`○, `kusner2017counterfactual`○, `kilbertus2017avoiding`○, `zemel2013learning`○
- Review methodology (§3.1): `kitchenham2007guidelines`○, `page2021prisma`○, `wohlin2014guidelines`○

### Evaluation (every section's table, Ch. 4)
- **Seeds:** `pawelczyk2021carla`○ (CARLA benchmark), `wexler2020whatif` (What-If Tool), `lequy2022survey`○, `becker1996adult`○, `hofmann1994statlog`○, `angwin2016machine`○, `wightman1998lsac`○, `yeh2009comparisons`○
- **Must answer:** Which datasets have sensitive attributes, actionable features and a known or assumed causal graph? Which metrics are reported (validity, cost, sparsity, plausibility, recourse-fairness gap), and with which statistical tests?
- **Deliverable:** **datasets table** (domain | n | features | sensitive attr. | actionable features | causal graph? | used by), placed in Ch. 4 (experimental design); datasets and metrics columns in every Ch. 3 table.

---

## 6. Week-by-week timeline

| Week | Dates | Reading / searching | Writing | Checkpoint |
|---|---|---|---|---|
| W1 | 27 Oct – 1 Nov | Set up Zotero + BBT. Reorganise the spreadsheet into G1–G4, Foundations and Evaluation; fill missing author/year fields; log the remaining 5 G3 seeds. Skim the anchor surveys (`karimi2022survey`, `verma2024counterfactual`, `liu2025kan`). | One-page scope note and draft RQs | **Meeting with supervisors** (slides in `presentation/lit_review_plan/`): grouping, scope (RL work in or out), RQs, delivery date, whether preliminary experiments are expected |
| W2 | 2 – 8 Nov | Finalize strings, run S1–S4, deduplicate, log counts, screen titles/abstracts | §2.1 notation (start fixing notation early) | — |
| W3 | 9 – 15 Nov | Full-text screening + snowballing; fill the spreadsheet; fill the Evaluation tab (datasets, metrics, CARLA) | §3.1 review method; §2.2 KAN fundamentals | — |
| W4 | 16 – 22 Nov | **G1** counterfactual search | §2.5 + §3.2 + search-methods table | — |
| W5 | 23 – 29 Nov | **G2**, part 1: data manifold, causal constraints, imperfect causal knowledge | §2.4 + §3.3.1–3.3.3 | — |
| W6 | 30 Nov – 6 Dec | **G2**, part 2: deep causal generative models | §3.3.4–3.3.5 + plausibility and causal models table | — |
| W7 | 7 – 13 Dec | **G3** fairness of recourse | §2.3 + §3.4 + notions table | **Send Ch. 2 draft** |
| W8 | 14 – 20 Dec | **G4** KAN components; re-check the gap; freeze the search (~20 Dec) | §3.5 + KAN uses table; **§3.6 synthesis and gaps → final RQs**; PRISMA diagram and counts in §3.1 | **Send full Ch. 3 draft before the break** |
| W9–10 | 21 Dec – 3 Jan | Holiday buffer (plan for ~50% capacity) | Integrate feedback. Ch. 4 draft: problem formulation, architecture decision matrix, experimental design with the datasets table. *Optional:* sanity experiment (KAN vs MLP on Adult, baseline recourse, cost gap between groups) only if the supervisors expect preliminary work | — |
| W11 | 4 – 10 Jan | — | Ch. 1 (context, RQs, Gantt for 1st + 2nd semester, risk table, structure); Ch. 5; abstracts EN/PT; keywords; acronyms; AI declaration | — |
| W12 | 11 – 17 Jan | Last alert triage | Full revision: notation consistency, every claim cited, figures, bib clean-up (published versions over arXiv), proofreading | **Send complete draft ~13 Jan** |
| W13 | 18 – 22 Jan | — | Final fixes, final PDF | **Submit** |

**Every week:** Monday alert triage (30 min). Friday: push the LaTeX and the extraction spreadsheet to the repo, and send the supervisors a 3-bullet update (done / next / blockers).

---

## 7. Quality bar for Chapter 3 (taken from the example reports)

- **Synthesize, don't catalogue.** Group papers by idea, compare them and criticize them. Give key papers one paragraph at most, and cover minor papers together in one sentence.
- **End every section with a "Summary and implications" subsection** (Cruz's style) that ties back to the thesis and to options O1–O3.
- **Comparison tables are the backbone** (Mercier's style): search methods (G1), plausibility and causal models (G2), fairness-of-recourse notions (G3), KAN uses (G4), each with datasets and metrics columns. The datasets table goes in Ch. 4. Ch. 4 reuses them to justify choices.
- **Back every gap claim** with the search log and the intersection counts.
- **Define notation once** in Ch. 2 and reuse it everywhere.
- **Planned figures:** pipeline figure with G1–G4, search-strategy taxonomy, PRISMA flow diagram, KAN vs MLP schematic, causal recourse example (SCM + intervention).
- **Prefer peer-reviewed versions** over arXiv when both exist.

---

## 8. Risks to the literature review and mitigations

| Risk | Mitigation |
|---|---|
| Huge KAN literature, mostly application papers | Screen S4 only. Review application-only papers as a count, not individually. |
| G2 is the largest group (data manifold + causal recourse + deep causal models) | It gets two weeks (W5–W6). |
| W8 combines G4, the search freeze and the synthesis | G4 builds on §2.2 (written in W3) and has the fewest papers. If W8 overruns, move §3.6 synthesis to W9. |
| G3 is thin in the spreadsheet (3 of 28 papers) | Log the remaining 5 seeds in W1; S2 targets it directly. |
| Field moves fast; the gap gets filled by a new preprint | Weekly alerts. If scooped, shift the angle (e.g., KAN interpretability for *auditing* recourse unfairness, or a different fairness notion). |
| Scope creep into general XAI/fairness or RL | Apply IC1–IC2 strictly, settle the RL scope in W1, and use the background chapter for general material. |
| Christmas break | W9–10 are buffer weeks, and Ch. 3 is due before the break. |
| `references.bib` drifts from the spreadsheet | Add each paper to the spreadsheet and `references.bib` at the same time; Zotero + BBT as the single source of truth; DBLP check in W12. |
| Architecture undecided at delivery | Acceptable for an intermediate report, as long as Ch. 4 presents the decision matrix and the recommended option. |
