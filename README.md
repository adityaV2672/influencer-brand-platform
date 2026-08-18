# AI-Powered Influencer–Brand Collaboration Platform

A creator-discovery and brand-matching platform: it scores social-media creators
on predicted **sponsored-campaign performance** rather than follower count, matches
them to brand briefs with semantic similarity and hard brand-safety gates, estimates
a collaboration price band, and serves all of it through a freemium dashboard.

---

## What this project actually demonstrates

Three findings, all measured rather than asserted:

1. **Word-list sentiment methods fail on irony — measurably.** On the TweetEval irony
   corpus, every lexicon method (Bing, VADER, NRC) scores at or below the
   majority-class baseline. Sarcasm inverts meaning without changing vocabulary, so
   the signal a word list needs is simply not present.
2. **NRC is *worse* than VADER at polarity**, despite being the richer lexicon. Its
   value is the eight emotion categories, not the positive/negative split. The system
   uses it accordingly.
3. **The machine-learned price model barely beats a published rate card.** That is a
   negative result with a clear consequence: ship the rule, skip the model, revisit
   when real deal data exists.

The scoring model reaches ~80% of the theoretically achievable R² and substantially
beats both the published benchmark curve and the hand-weighted composite index that
the original project design proposed.

---

## Honest description of the data

| Component | Status |
|---|---|
| 2,000 creators, their posts, metrics, and campaign outcomes | **Synthetic** |
| Engagement-rate and INR fee calibration | Fitted to **published 2026 industry benchmarks**; every tier median verified inside the published band |
| All sentiment / emotion / irony accuracy figures | Measured on **real, human-labelled corpora** (TweetEval, Misra & Arora sarcasm headlines) |
| Model metrics | Out-of-fold under **GroupKFold on creator id**, leakage-checked in code |

This is a **simulation study**. Creators are generated from hidden latent traits
(content quality, audience authenticity, consistency, ad saturation); observable
features are noisy functions of those traits, and the model never sees them. Because
the noise level is set by construction, the maximum achievable R² is *known*, and
model performance is reported as a fraction of it rather than as a bare number.

**Why the NLP evaluation uses real data instead.** Synthetic text cannot validate an
NLP method: measuring how well a detector recovers a label we injected only measures
how well it reverse-engineers our own template. So every NLP claim in this project is
measured on text written by real people and labelled by real annotators.

---

## Quick start

```bash
# 1. environment
conda create -n influencer python=3.11 -y
conda activate influencer
pip install -r requirements-dev.txt

# 2. build every artifact (~45-90 min; transformers run over 52k posts)
python run_pipeline.py

# 3. run the dashboard
streamlit run app/Home.py
```

Stages are individually runnable:

```bash
python run_pipeline.py --list             # show all stages
python run_pipeline.py --only nlp models  # re-run just these
python run_pipeline.py --skip evaluate    # skip the NLP benchmark suite
```

### Optional: LLM sarcasm analysis

The LLM methods need a local [Ollama](https://ollama.com) instance. No API key, no
cost, no data leaves the machine.

```bash
ollama pull qwen2.5:7b-instruct
python -m src.benchmark.run_benchmarks --tasks irony      # includes LLM rows
```

Without Ollama the pipeline runs normally and records the LLM rows as `skipped`
rather than silently omitting them.

---

## Pipeline

| Stage | What it does | Depends on |
|---|---|---|
| `generate` | Synthetic creator universe: profiles, posts, brands | — |
| `benchmarks` | Downloads the real labelled NLP corpora | — |
| `sna` | Builds the creator graph, computes centrality and communities | `generate` |
| `campaigns` | Simulates sponsored campaigns — the supervised target | `sna` |
| `nlp` | Sentiment, emotion, embeddings, topics, irony over every post | `generate` |
| `features` | Merges all tracks into the influencer feature table | all above |
| `models` | Trains the performance and price models | `features` |
| `brandfit` | Semantic brand-fit matrix with safety gates | `features` |
| `evaluate` | Benchmarks every NLP method on the real corpora | `benchmarks` |
| `export` | Writes the slim artifact bundle the dashboard reads | all above |

**Stage ordering is load-bearing.** `campaigns` runs *after* `sna` because campaign
outcomes depend on measured topical centrality. An earlier version used a proxy, which
made the network features statistically independent of the target and the entire SNA
pillar decorative.

---

## Architecture

```
Raw data ──┬── CONTENT  → NLP pipeline (VADER · NRC · SBERT · BERTopic · RoBERTa · LLM)
           ├── PROFILE  → reach, engagement, growth features
           └── NETWORK  → co-hashtag / co-brand graph → centrality, communities
                                    │
                        influencer feature table
                                    │
                ┌───────────────────┼───────────────────┐
        performance model     brand-fit composite   price model
         (LightGBM)          (SBERT + gates)        (LightGBM)
                                    │
                          freemium dashboard
```

**Offline scoring / online serving.** Every heavy model runs once in the pipeline and
is cached to disk. The deployed dashboard loads **no ML model at all** — it reads
precomputed Parquet from `app_data/`. That is what keeps the hosted app inside a ~1 GB
free tier, and it is the same split production recommender systems use.

This is why there are two requirements files:

- `requirements.txt` — dashboard runtime only (no torch, no transformers)
- `requirements-dev.txt` — the full pipeline

---

## Layout

```
src/
  config.py                 all paths, constants, hyper-parameters
  data/
    benchmarks.py           published industry benchmarks, with provenance
    generate_synthetic.py   the simulation
    lexicon.py              niche vocabulary for caption generation
    fetch_benchmarks.py     downloads the real labelled corpora
  nlp/
    base.py                 one interface every method implements
    lexicons.py             Bing · VADER · NRC
    embeddings.py           SBERT + supervised head
    transformers_hf.py      CardiffNLP RoBERTa checkpoints
    sarcasm.py              LLM prompting (zero-shot / few-shot / chain-of-thought)
    topics.py               BERTopic vs LDA, with coherence scoring
    extract.py              hashtags, brands, promo language, CTAs
    pipeline.py             runs everything over the post corpus
  network/sna.py            graph construction and centrality
  features/                 feature store, leakage controls, dashboard export
  models/                   performance, price, brand-fit
  benchmark/                the evaluation harness
  report/                   figures and the Word report
app/                        the Streamlit dashboard
app_data/                   deployment payload (committed on purpose)
```

---

## Evaluation discipline

The harness enforces these, because the comparison is worthless otherwise:

- **Identical rows for every method.** Where a slow method is subsampled, the fast
  methods are additionally scored on that same subsample.
- **GroupKFold on creator id.** A creator contributes up to three campaigns; a random
  split would place the same creator in train and test and inflate every metric.
- **A majority-class baseline is always present.** A method that cannot beat "always
  guess the most common label" has demonstrated nothing.
- **Throughput recorded.** A method 200× slower for +2 F1 is a real trade-off.
- **Leakage gated in code.** A banned-substring list guards the model matrix and the
  trainer raises if anything slips through.
- **Missing dependencies produce a recorded SKIP row**, never a silent omission.

---

## Known limitations

- The network graph is **topical, not social** — Instagram exposes no follower edges
  to third parties. PageRank here means embeddedness in a shared vocabulary.
- Betweenness centrality is a sampled approximation (Brandes–Pich).
- **The content pillar shows no measurable contribution** to performance prediction in
  the ablation. It earns its place through brand-safety screening and explanation, not
  prediction, and the report says so.
- The price model's high R² is partly circular — fees are generated from features the
  model can see.
- Industry benchmark sources are marketing publications, not peer-reviewed research.

---

## Citations

- Barbieri, F. et al. (2020). *TweetEval: Unified Benchmark and Comparative Evaluation
  for Tweet Classification.* Findings of EMNLP. https://aclanthology.org/2020.findings-emnlp.148/
- Misra, R. & Arora, P. *Sarcasm Detection using News Headlines Dataset.* arXiv:2212.06035
- Hutto, C.J. & Gilbert, E. (2014). *VADER.* ICWSM.
- Mohammad, S.M. & Turney, P.D. (2013). *Crowdsourcing a Word-Emotion Association
  Lexicon.* Computational Intelligence 29(3).
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT.* EMNLP.
- Grootendorst, M. (2022). *BERTopic.* arXiv:2203.05794
- Hu, M. & Liu, B. (2004). *Mining and Summarizing Customer Reviews.* KDD.
- Blondel, V. et al. (2008). *Fast unfolding of communities in large networks.* J. Stat. Mech.
- Ke, G. et al. (2017). *LightGBM.* NeurIPS.
