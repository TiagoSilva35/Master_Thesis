# Literature Review Plan: Fair Algorithmic Recourse with Kolmogorov-Arnold Networks

**Author:** Tiago Silva · MSc in Informatics Engineering, specialization in Artificial Intelligence (DEI, University of Coimbra)
**Supervisors:** Prof. Pedro Henriques Abreu (DEI, University of Coimbra) · Prof. Bruno Veloso (INESC TEC; Faculty of Economics, University of Porto)
**Intermediate delivery:** January 2027. The plan assumes **~22 Jan 2027**; confirm the official date and shift the last two weeks if needed.
**Plan start:** Tue 27 Oct 2026, which gives about 13 weeks including the Christmas break.

---

## 0. What the January delivery must contain

I based this on the three DEI reports supervised by Prof. Abreu that were provided as examples: Mercier 2018 (intermediate report), Cortesão 2024 and Cruz Jan-2026 (the closest model: same supervisor, same specialization, same delivery point).

| Template file | Chapter | Content | Target pages |
|---|---|---|---|
| `text/1-introduction.tex` | 1 Introduction | Context, objectives/RQs, **work plan (Gantt, 1st sem. actual vs planned + 2nd sem.)**, **risk analysis table**, document structure | 5–7 |
| `text/2-background.tex` | 2 Background Knowledge | Self-contained theory: notation, KANs, fairness, causality, counterfactuals/recourse | 15–18 |
| `text/3-literature-review.tex` | 3 Literature Review | **The core of the delivery.** Review method, method taxonomies, comparison tables, "summary and implications" per section, research gaps | 25–32 |
| `text/4-proposed-approach.tex` | 4 Proposed Approach | Problem formulation, candidate architectures (decision matrix), experimental design | 5–8 |
| `text/5-conclusion.tex` | 5 Conclusion | First-semester summary, next steps | 1–2 |

Overall target: **~55–70 pages of main text and 100–150 references.** Cruz's report has about 65 pages of text plus 24 pages of references.

Frontmatter to fill in during the final weeks (template files, format unchanged): abstract EN/PT, keywords, acronyms (`frontmatter/8-acronyms.tex`), acknowledgements and the **AI-use declaration** (`frontmatter/5-ai-declaration.tex`). Record AI-tool use as you go so the declaration is accurate.

---

## 1. Scope: four pillars, one intersection

```
         (A) Counterfactual explanations
             & algorithmic recourse
                     │
   (B) Algorithmic ──┼── (D) Kolmogorov-Arnold
       fairness      │       Networks
                     │
         (C) Causal modelling (SCMs) ── enabler for causally valid recourse

   Thesis = A ∩ B ∩ D, with C as needed machinery
```

**Early gap check (27 Oct 2026, quick web search; must be confirmed by the systematic search):**
- No work combining KANs with counterfactual explanations or algorithmic recourse turned up.
- KANs and fairness: only *Learning Fair Representations with KANs* (Priyadarshini & Gago-Masague, AAAI-26) appeared. A KAN imbalanced-data study (Yadav & Vijay, 2025) explicitly states that KAN bias mitigation on tabular data is unvalidated.
- Fair recourse is active right now. *Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden* (Barrainkua et al., AAAI-26) is the latest seed.

### The review must decide the architecture
You are not yet sure of the architecture, so Chapter 3 should be written to **end in a decision**. Each candidate role for the KAN raises its own questions:

| Option | Role of the KAN | What the review must answer |
|---|---|---|
| **O1. KAN as the decision model** | Classifier whose additive univariate structure makes recourse cheaper to compute and easier to explain. A fairness-of-recourse term is added during training. | How do recourse methods exploit model structure (linear/MIP, gradients)? Which in-training recourse-fairness regularizers exist? Are KANs competitive on tabular data? |
| **O2. KAN as the causal model** | KANs parameterize the structural equations of an SCM used to compute causally consistent recourse (interventions, not just feature changes). | Which causal generative models exist, and what are their identifiability guarantees and data types? How is causal recourse fairness defined (individual vs group)? Where does the causal graph come from? |
| **O3. KAN as the recourse generator** | Amortized model that maps an individual to a recommended action, with fairness constraints. | Which amortized/generative recourse methods exist? How do they handle actionability and plausibility? |
| **O4. Combination** | Any of O1–O3 together | Trade-offs in complexity and evaluability |

Chapter 4 then justifies the choice with this table, filled in from the evidence.

---

## 2. Preliminary research questions (refine with the supervisors in W1, freeze in W8)

- **RQ1.** How can KANs be integrated into a recourse pipeline (as decision model, causal model or generator) so that recommendations are valid, actionable and interpretable?
- **RQ2.** Which fairness-of-recourse notions are meaningful for KAN-based models (group cost parity, social burden, individual/causal fairness of recourse), and how can they be measured?
- **RQ3.** Does the interpretability of KANs (univariate edge functions, pruning, symbolic formulas) help to *audit and explain* recourse disparities better than MLP-based counterparts?
- **RQ4.** What is the trade-off between predictive performance, recourse cost and recourse fairness?

---

## 3. Search protocol (semi-systematic)

None of the example reports used a formal protocol. A documented protocol is still cheap to add (1–2 pages, §3.1). It makes your gap claims defensible ("no study in our search combined X and Y"). Follow Kitchenham & Charters (2007) for the protocol, PRISMA 2020 for the flow diagram and Wohlin (2014) for snowballing.

### 3.1 Sources
- **Databases:** Scopus, Web of Science, ACM Digital Library, IEEE Xplore, DBLP.
- **Preprints (mandatory here):** arXiv, Semantic Scholar, Google Scholar. The KAN literature is from 2024 onwards and is mostly preprints.
- **Venues to scan manually (2019–2026):** FAccT, AIES, NeurIPS, ICML, ICLR, AAAI, IJCAI, AISTATS, UAI, CLeaR, KDD, TMLR, JMLR. Journals: ACM Computing Surveys, Data Mining and Knowledge Discovery, Artificial Intelligence, Machine Learning, IEEE TNNLS.

### 3.2 Search strings (title/abstract/keywords; adapt syntax per database)

| ID | Cluster | String |
|---|---|---|
| S1 | Recourse / CE | `("algorithmic recourse" OR "actionable recourse" OR "counterfactual explanation*" OR "counterfactual recourse")` |
| S2 | Fair recourse | `S1 AND (fair* OR "equal* recourse" OR "recourse disparit*" OR "social burden" OR discriminat* OR bias)` |
| S3 | Causal recourse / counterfactual inference | `("causal recourse" OR ("recourse" AND "structural causal model*") OR ("counterfactual inference" AND ("normalizing flow*" OR "variational" OR "diffusion" OR "neural causal model*")))` |
| S4 | KAN core | `("Kolmogorov-Arnold network*" OR "Kolmogorov Arnold network*")` |
| S5 | KAN ∩ trustworthy ML | `S4 AND (interpretab* OR explainab* OR "symbolic regression" OR fair* OR counterfactual OR causal OR recourse OR tabular)` |

S4 will return thousands of application papers. Screen **S5** fully, and use S4 only for surveys and the most-cited methodological variants.

### 3.3 Inclusion and exclusion criteria
- **IC1** Proposes or evaluates a method, metric, benchmark or theoretical result on CE/recourse, fairness of recourse, causal counterfactual inference, or KAN methodology.
- **IC2** Applicable to tabular data.
- **IC3** Peer-reviewed. A preprint qualifies only if it is highly cited, or directly on-topic and less than 12 months old.
- **IC4** Written in English.
- **Time windows:** recourse/CE 2017–2026 (from Wachter et al.); KANs 2024–2026, plus the classical theorem papers; fairness uses foundational papers (2012–2017), surveys, and recourse-related work from 2019 on.
- **EC1** Image/text-only CE with no transfer to tabular data.
- **EC2** "Recourse" in the legal or operations-research sense.
- **EC3** KAN application papers with no methodological contribution (count them, do not review them).
- **EC4** Abstracts, posters or theses.
- **EC5** Full text unavailable.

### 3.4 Screening pipeline (keep a log of counts for the PRISMA diagram!)
1. Run S1–S5 in every source, export to Zotero, deduplicate, and **record counts per source**.
2. Screen by title/abstract (fast: include / exclude / maybe).
3. Screen the full text of the includes and maybes, and fill the extraction matrix (§4).
4. Snowball: one backward and one forward round from the seed set (`references.bib`) and the strongest includes.
5. **Freeze the systematic search around 20 Dec 2026.** Papers found later enter only as "recent work" if crucial.

---

## 4. Tooling and note-taking workflow

- **Zotero + Better BibTeX** auto-exports `references.bib` with citekeys in the same style as the seed file (`karimi2021algorithmic`).
  - One collection per cluster (A–E).
  - Tags: `screen:in`, `screen:maybe`, `read:skim`, `read:deep`, `rel:1/2/3`.
- **Extraction matrix** (one spreadsheet, one row per included paper):
  `citekey | year | venue | cluster | problem | method/model | model class (linear/tree/MLP/KAN) | causal assumption (none / known SCM / partial graph / learned) | fairness notion (group / individual / causal / burden) | recourse cost definition | actionability handling | datasets | metrics | code? | limitations | relevance (1–3)`
- **Five-line note per deep-read paper:** problem, method, assumptions, key result, limitation/relevance to me. These notes become paragraphs almost directly.
- **Snowballing:** Semantic Scholar "cited by", ResearchRabbit or Connected Papers.
- **Alerts:** Google Scholar alerts for `"Kolmogorov-Arnold" recourse`, `"Kolmogorov-Arnold" fairness` and `"algorithmic recourse" fairness`, plus arXiv cs.LG keyword alerts. Spend **30 minutes every Monday** triaging them.
- **Write-as-you-read rule:** each cluster week ends with that section drafted in LaTeX, even if rough.

---

## 5. Reading clusters, seed papers and must-answer questions

All seeds are in `references.bib`. **Verify the volume, pages and venue of every entry against DBLP or the publisher before final submission.**

### Cluster A: Counterfactual explanations and recourse methods (§2.5, §3.2)
- **Seeds:**
  - Definitions and surveys: `wachter2018counterfactual`, `karimi2022survey`, `verma2024counterfactual`, `guidotti2024counterfactual`
  - Methods: `ustun2019actionable`, `mothilal2020explaining`, `poyiadzi2020face`, `joshi2019towards`, `pawelczyk2020learning`, `mahajan2019preserving`, `upadhyay2021towards`, `rawal2020beyond`
  - Benchmarking: `pawelczyk2021carla`
  - Critical perspectives: `venkatasubramanian2020philosophical`, `barocas2020hidden`
- **Must answer:** What taxonomy will you use (model access × search strategy × plausibility mechanism)? Which desiderata and metrics are standard? How do methods exploit model structure? Which methods are amortized (relevant to O3)?
- **Deliverable:** taxonomy figure and the **recourse-methods comparison table**.

### Cluster B: Fairness and fairness of recourse (§2.3, §3.3)
- **Seeds:**
  - General fairness: `mehrabi2021survey`, `caton2024fairness`, `barocas2023fairness`, `dwork2012fairness`, `hardt2016equality`, `kleinberg2017inherent`, `chouldechova2017fair`, `kusner2017counterfactual`, `kilbertus2017avoiding`, `zemel2013learning`
  - Fairness of recourse: `gupta2019equalizing`, `sharma2020certifai`, `vonkugelgen2022fairness`, `ehyaei2023robustness`, `kavouras2023fairness`, `bell2024fairness`, `barrainkua2026revisiting`
- **Must answer:** Which recourse-fairness notions exist (cost parity, burden, individual/causal fairness of recourse, substantive equality), and which require an SCM? Are there in-training mitigation methods (O1) or only post-hoc audits? How do fair classifiers interact with fair recourse (the Barrainkua et al. finding)?
- **Deliverable:** **fairness-of-recourse notions table** (notion | definition | needs SCM? | mitigation? | datasets).

### Cluster C: Causal modelling for recourse (§2.4, §3.4)
- **Seeds:**
  - Foundations: `pearl2009causality`, `peters2017elements`
  - Causal recourse: `karimi2021algorithmic`, `karimi2020algorithmic`, `dominguezolmedo2022adversarial`
  - Causal generative models: `pawlowski2020deep`, `khemakhem2021causal`, `sanchezmartin2022vaca`, `javaloy2023causal`, `chao2023interventional`, `xia2023neural`
- **Must answer:** What is the difference between counterfactual explanations and causal recourse? What happens when the SCM is unknown or misspecified? Which deep causal models support counterfactual (rung-3) queries, under which assumptions, and on mixed-type data? Which are interpretable?
- **Deliverable:** **causal models table** (family | identifiability | mixed data? | interpretable mechanisms?).

### Cluster D: Kolmogorov-Arnold Networks (§2.2, §3.5)
- **Seeds:**
  - Theory and core model: `kolmogorov1957representation`, `arnold1957functions`, `girosi1989representation`, `liu2025kan`, `liu2024kan`
  - Comparisons and variants: `yu2024kan`, `li2024kolmogorov`
  - Surveys: `somvanshi2024survey`, `hou2024comprehensive`
  - Tabular data, fairness and imbalance: `poeta2024benchmarking`, `priyadarshini2026learning`, `yadav2025kolmogorov`
  - Related interpretable additive models: `agarwal2021neural`, `rudin2019stop`
- **Must answer:** Which variants exist (B-spline, RBF, Chebyshev, wavelet, Fourier, ...) and what are their trade-offs? Do KANs actually beat MLPs on tabular data under fair comparison? What evidence supports their interpretability claims? Which works use KANs for fairness, XAI or causality, i.e. **confirm the gap**?
- **Deliverable:** **KAN variants and uses table**, and the explicit gap statement.

### Cluster E: Datasets, metrics and benchmarking practice (§3.6)
- **Seeds:** `lequy2022survey`, `becker1996adult`, `hofmann1994statlog`, `angwin2016machine`, `wightman1998lsac`, `yeh2009comparisons`, `pawelczyk2021carla`
- **Must answer:** Which datasets have sensitive attributes, actionable features and a known or assumed causal graph? Which metrics are reported (validity, cost, sparsity, plausibility, recourse-fairness gap), and with which statistical tests?
- **Deliverable:** **datasets table** (domain | n | features | sensitive attr. | actionable features | causal graph? | used by).

---

## 6. Week-by-week timeline

| Week | Dates | Reading / searching | Writing | Checkpoint |
|---|---|---|---|---|
| W1 | 27 Oct – 1 Nov | Set up Zotero + BBT; add the full template (`frontmatter/`, `text/9-appendices.tex`, images) to the repo so it compiles anywhere. Skim the anchor surveys (`karimi2022survey`, `verma2024counterfactual`, `mehrabi2021survey`, `liu2025kan`). | One-page scope note and draft RQs | **Meeting with supervisors:** scope, RQs, clusters, delivery date, whether preliminary experiments are expected |
| W2 | 2 – 8 Nov | Finalize strings, run S1–S5, deduplicate, log counts, screen titles/abstracts | §2.1 notation (start fixing notation early) | — |
| W3 | 9 – 15 Nov | Full-text screening + snowballing; fill the matrix | §3.1 review method; §2.2 KAN fundamentals | — |
| W4 | 16 – 22 Nov | **Cluster A** deep read | §2.5 + §3.2 + recourse table | — |
| W5 | 23 – 29 Nov | **Cluster B** deep read | §2.3 + §3.3 + fairness table | **Send Ch. 2 draft** |
| W6 | 30 Nov – 6 Dec | **Cluster C** deep read | §2.4 + §3.4 + causal models table | — |
| W7 | 7 – 13 Dec | **Cluster D** deep read; re-check the gap | §3.5 + KAN table | — |
| W8 | 14 – 20 Dec | **Cluster E**; freeze the search (~20 Dec) | §3.6 datasets table; **§3.7 synthesis and gaps → final RQs**; PRISMA diagram | **Send full Ch. 3 draft before the break** |
| W9–10 | 21 Dec – 3 Jan | Holiday buffer (plan for ~50% capacity) | Integrate feedback. Ch. 4 draft: problem formulation, architecture decision matrix, experimental design. *Optional:* sanity experiment (KAN vs MLP on Adult, baseline recourse, cost gap between groups) only if the supervisors expect preliminary work | — |
| W11 | 4 – 10 Jan | — | Ch. 1 (context, RQs, Gantt for 1st + 2nd semester, risk table, structure); Ch. 5; abstracts EN/PT; keywords; acronyms; AI declaration | — |
| W12 | 11 – 17 Jan | Last alert triage | Full revision: notation consistency, every claim cited, figures, bib clean-up (published versions over arXiv), proofreading | **Send complete draft ~13 Jan** |
| W13 | 18 – 22 Jan | — | Final fixes, final PDF | **Submit** |

**Every week:** Monday alert triage (30 min). Friday: push the LaTeX and the extraction matrix to the repo, and send the supervisors a 3-bullet update (done / next / blockers).

---

## 7. Quality bar for Chapter 3 (taken from the example reports)

- **Synthesize, don't catalogue.** Group papers by idea, compare them and criticize them. Give key papers one paragraph at most, and cover minor papers together in one sentence.
- **End every section with a "Summary and implications" subsection** (Cruz's style) that ties back to the thesis and to options O1–O3.
- **Comparison tables are the backbone** (Mercier's style): recourse methods, fairness notions, causal models, KAN variants, datasets. Ch. 4 reuses them to justify choices.
- **Back every gap claim** with the search log.
- **Define notation once** in Ch. 2 and reuse it everywhere.
- **Planned figures:** positioning diagram (the four pillars), recourse taxonomy, PRISMA flow diagram, KAN vs MLP schematic, causal recourse example (SCM + intervention).
- **Prefer peer-reviewed versions** over arXiv when both exist.

---

## 8. Risks to the literature review and mitigations

| Risk | Mitigation |
|---|---|
| Huge KAN literature, mostly application papers | Screen S5 only. Review application-only papers as a count, not individually. |
| Field moves fast; the gap gets filled by a new preprint | Weekly alerts. If scooped, shift the angle (e.g., KAN interpretability for *auditing* recourse unfairness, or a different fairness notion). |
| Scope creep into general XAI/fairness | Apply IC1–IC2 strictly, and use the background chapter for general material. |
| Christmas break | W9–10 are buffer weeks, and Ch. 3 is due before the break. |
| Bibliography errors | Zotero + BBT as the single source of truth; DBLP check in W12. |
| Architecture undecided at delivery | Acceptable for an intermediate report, as long as Ch. 4 presents the decision matrix and the recommended option. |
