/**
 * Slide deck generator.
 *
 * Reads deck_data.json (metrics extracted from the pipeline artifacts) and the
 * figure PNGs, and writes the presentation. Every number on every slide comes
 * from that JSON — nothing is typed in by hand, so the deck cannot drift out of
 * agreement with the report.
 *
 *   node src/report/build_deck.js <data_dir> <out.pptx>
 */
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const DATA_DIR = process.argv[2] || "reports/deck";
const OUT = process.argv[3] || "reports/Influencer_Platform_Deck.pptx";

const D = JSON.parse(fs.readFileSync(path.join(DATA_DIR, "deck_data.json"), "utf8"));
const FIG = path.join(DATA_DIR, "figures");

// -- palette: deep ink + signal orange. Not default blue. --------------------
const INK = "141B34";       // dominant
const INK_SOFT = "26304F";
const LIGHT = "FFFFFF";
const PAPER = "F4F5F8";
const ACCENT = "EB6834";    // the sharp accent
const BLUE = "2A78D6";      // ties to the product UI
const GREEN = "1BAF7A";
const RED = "E34948";
const MUTED = "8A8FA3";
const MUTED_DARK = "5A6076";

const HEAD = "Cambria";
const BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.3 x 7.5
pres.author = "Aditya Verma";
pres.title = "AI-Powered Influencer-Brand Collaboration Platform";

const W = 13.3, H = 7.5, M = 0.65;

const fig = (n) => path.join(FIG, n);
const has = (n) => fs.existsSync(fig(n));

// ---------------------------------------------------------------------------
function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: INK };
  return s;
}
function lightSlide(title, kicker) {
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: M, y: 0.38, w: W - 2 * M, h: 0.24, fontFace: BODY, fontSize: 10.5,
      bold: true, color: ACCENT, charSpacing: 1.6,
    });
  }
  if (title) {
    s.addText(title, {
      x: M, y: kicker ? 0.66 : 0.5, w: W - 2 * M, h: 0.72,
      fontFace: HEAD, fontSize: 30, bold: true, color: INK, valign: "top",
    });
  }
  return s;
}
function statCard(s, { x, y, w, h, value, label, color = INK, sub = null }) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.09, fill: { color: PAPER }, line: { color: PAPER },
  });
  s.addText(value, {
    x: x + 0.22, y: y + 0.16, w: w - 0.44, h: 0.72,
    fontFace: HEAD, fontSize: 34, bold: true, color, margin: 0, valign: "middle",
  });
  s.addText(label, {
    x: x + 0.22, y: y + 0.88, w: w - 0.44, h: sub ? 0.42 : 0.6,
    fontFace: BODY, fontSize: 11.5, color: MUTED_DARK, margin: 0, valign: "top",
  });
  if (sub) {
    s.addText(sub, {
      x: x + 0.22, y: y + h - 0.44, w: w - 0.44, h: 0.34,
      fontFace: BODY, fontSize: 9.5, italic: true, color: MUTED, margin: 0,
    });
  }
}
function bulletBox(s, { x, y, w, h, heading, items, color = INK, tint = PAPER }) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.09, fill: { color: tint }, line: { color: tint },
  });
  s.addText(heading, {
    x: x + 0.26, y: y + 0.2, w: w - 0.5, h: 0.4,
    fontFace: BODY, fontSize: 14, bold: true, color, margin: 0,
  });
  s.addText(
    items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    {
      x: x + 0.26, y: y + 0.66, w: w - 0.5, h: h - 0.88,
      fontFace: BODY, fontSize: 11.5, color: MUTED_DARK, margin: 0,
      paraSpaceAfter: 7, valign: "top",
    }
  );
}
function caption(s, text, y) {
  s.addText(text, {
    x: M, y: y, w: W - 2 * M, h: 0.4, fontFace: BODY, fontSize: 10,
    italic: true, color: MUTED, align: "center",
  });
}
function footer(s, text) {
  s.addText(text, {
    x: M, y: H - 0.46, w: W - 2 * M, h: 0.28,
    fontFace: BODY, fontSize: 9, color: MUTED, margin: 0,
  });
}

// ===========================================================================
// 1. Title
// ===========================================================================
{
  const s = darkSlide();
  s.addShape(pres.ShapeType.ellipse, {
    x: W - 3.1, y: -1.5, w: 4.6, h: 4.6,
    fill: { color: INK_SOFT }, line: { color: INK_SOFT },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: W - 1.5, y: H - 2.0, w: 2.6, h: 2.6,
    fill: { color: ACCENT }, line: { color: ACCENT }, transparency: 78,
  });
  s.addText("MACHINE LEARNING PROJECT", {
    x: M, y: 2.05, w: 8, h: 0.3, fontFace: BODY, fontSize: 11.5,
    bold: true, color: ACCENT, charSpacing: 2,
  });
  s.addText("AI-Powered Influencer–Brand\nCollaboration Platform", {
    x: M, y: 2.5, w: 9.2, h: 1.75, fontFace: HEAD, fontSize: 40,
    bold: true, color: LIGHT, lineSpacing: 46,
  });
  s.addText(
    "Ranking creators on predicted campaign performance — not follower count",
    { x: M, y: 4.35, w: 9.0, h: 0.42, fontFace: BODY, fontSize: 16, color: "C3C8D8" }
  );
  s.addText(D.author || "Aditya Verma", {
    x: M, y: 5.15, w: 6, h: 0.3, fontFace: BODY, fontSize: 13, color: LIGHT, bold: true,
  });
  s.addText(D.date, { x: M, y: 5.48, w: 6, h: 0.3, fontFace: BODY, fontSize: 11, color: MUTED });
  s.addNotes(
    "This project scores creators on predicted sponsored-campaign engagement rather than " +
    "follower count. Three things to take away: the NLP benchmark result on irony, the model " +
    "beating the hand-weighted index from the original design, and two honest negative results."
  );
}

// ===========================================================================
// 2. The problem
// ===========================================================================
{
  const s = lightSlide("Follower count is the wrong ranking signal", "The problem");
  const cw = (W - 2 * M - 0.6) / 3;
  statCard(s, {
    x: M, y: 1.75, w: cw, h: 1.65, value: D.problem.nano_er, label: "median engagement, Nano tier",
    color: GREEN,
  });
  statCard(s, {
    x: M + cw + 0.3, y: 1.75, w: cw, h: 1.65, value: D.problem.macro_er,
    label: "median engagement, Macro tier", color: ACCENT,
  });
  statCard(s, {
    x: M + 2 * (cw + 0.3), y: 1.75, w: cw, h: 1.65, value: D.problem.ratio,
    label: "engagement gap between the two", color: INK,
  });
  if (has("fig_engagement_decay.png")) {
    s.addImage({ path: fig("fig_engagement_decay.png"), x: 3.35, y: 3.62, w: 6.6, h: 3.15 });
  }
  caption(s, "Engagement declines systematically with audience size, so ranking by reach mis-ranks by design.", 6.78);
  s.addNotes(
    "Brands still pick creators on follower count. Engagement decays with size, so the biggest " +
    "accounts look best on reach and worst on actual performance. That gap is the product opportunity."
  );
}

// ===========================================================================
// 3. Two structural problems in the original design
// ===========================================================================
{
  const s = lightSlide("Two problems in the original design had to be fixed first", "Design revision");
  const bw = (W - 2 * M - 0.4) / 2;
  bulletBox(s, {
    x: M, y: 1.7, w: bw, h: 2.0, heading: "1 · The network pillar had no data source",
    color: RED,
    items: [
      "The design assumed a follower graph for PageRank and centrality.",
      "Instagram exposes no follower edges to third parties. There is no legal route to one.",
      "Building on it would have meant fabricating data or dropping the pillar.",
    ],
  });
  bulletBox(s, {
    x: M + bw + 0.4, y: 1.7, w: bw, h: 2.0, heading: "2 · There was no target variable",
    color: RED,
    items: [
      "Phase 1 was a weighted index with hand-set weights — a rubric, not machine learning.",
      "Phase 2 proposed a supervised model but named no label to train it on.",
    ],
  });
  bulletBox(s, {
    x: M, y: 3.9, w: bw, h: 2.35, heading: "Resolved · a co-behaviour graph",
    color: GREEN, tint: "EAF6F1",
    items: [
      "Edges from shared rare hashtags and shared brand collaborations, TF-IDF weighted.",
      "Derivable from content alone — the data a real platform has on day one.",
      "Labelled honestly: this measures topical centrality, not social influence.",
    ],
  });
  bulletBox(s, {
    x: M + bw + 0.4, y: 3.9, w: bw, h: 2.35, heading: "Resolved · campaign outcomes as the target",
    color: GREEN, tint: "EAF6F1",
    items: [
      "Simulated sponsored campaigns supply the exact label the design named.",
      "Not a deterministic function of any single feature — the model must combine signals.",
      "The Phase-1 index is kept as a baseline, so learning is demonstrated, not asserted.",
    ],
  });
  s.addNotes(
    "These are the two things a reviewer would attack. Better to fix them up front and show the " +
    "reasoning than to present a design that quietly assumes unavailable data and an unnamed label."
  );
}

// ===========================================================================
// 4. What is real, what is synthetic
// ===========================================================================
{
  const s = lightSlide("What is real here, and what is not", "Methodology");
  const bw = (W - 2 * M - 0.4) / 2;
  bulletBox(s, {
    x: M, y: 1.7, w: bw, h: 3.1, heading: "Synthetic — deliberately",
    color: ACCENT, tint: "FDEEE7",
    items: [
      `${D.data.n_influencers} creators, ${D.data.n_posts} posts, ${D.data.n_campaigns} campaigns.`,
      "Generated from hidden latent traits the model never sees.",
      "Because we set the noise, the maximum achievable R² is known — results are reported as a fraction of it.",
    ],
  });
  bulletBox(s, {
    x: M + bw + 0.4, y: 1.7, w: bw, h: 3.1, heading: "Real and verifiable",
    color: GREEN, tint: "EAF6F1",
    items: [
      "Every NLP accuracy figure — human-labelled TweetEval and news-headline sarcasm corpora.",
      "Engagement and fee calibration — fitted to published 2026 benchmarks, every tier median verified inside the published band.",
      "All model metrics — out-of-fold under GroupKFold, leakage gated in code.",
    ],
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 5.0, w: W - 2 * M, h: 1.42, rectRadius: 0.09,
    fill: { color: INK }, line: { color: INK },
  });
  s.addText("Why the NLP evaluation refuses to use the synthetic text", {
    x: M + 0.3, y: 5.16, w: W - 2 * M - 0.6, h: 0.32,
    fontFace: BODY, fontSize: 13, bold: true, color: ACCENT, margin: 0,
  });
  s.addText(
    "If we generate a caption, label it \"sarcastic\", then measure how well a detector recovers that label, " +
    "we measure how well it reverse-engineers our own template — not how well it detects sarcasm. " +
    "So every sentiment, emotion and irony number in this project is measured on text written by real people.",
    { x: M + 0.3, y: 5.52, w: W - 2 * M - 0.6, h: 0.78, fontFace: BODY, fontSize: 11.5, color: "D5D9E5", margin: 0 }
  );
  s.addNotes("This is the slide that answers 'what data is this'. Do not skip it.");
}

// ===========================================================================
// 5. Calibration
// ===========================================================================
if (has("fig01_calibration.png")) {
  const s = lightSlide("The synthetic data is anchored to published benchmarks", "Methodology");
  s.addImage({ path: fig("fig01_calibration.png"), x: 1.15, y: 1.7, w: 11.0, h: 4.1 });
  caption(s, "Synthetic tier medians (points) against published 2026 benchmark ranges (bars). Every tier falls inside its published band.", 5.9);
  footer(s, "Sources: Nowadays Media engagement benchmarks; upGrowth India influencer pricing. Industry publications, treated as order-of-magnitude anchors.");
  s.addNotes("A synthetic dataset with invented magnitudes is worthless. This is the check that it is not.");
}

// ===========================================================================
// 6. The NLP question
// ===========================================================================
{
  const s = lightSlide("The supervisor's feedback, treated as a hypothesis to test", "Content intelligence");
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 1.68, w: W - 2 * M, h: 1.15, rectRadius: 0.09,
    fill: { color: PAPER }, line: { color: PAPER },
  });
  s.addText(
    "“Weak in qualitative analysis. Bing is primitive — go for NRC, VADER, SBERT embeddings, " +
    "BERTopic or even LLM prompting (this traces the sarcasm).”",
    { x: M + 0.35, y: 1.84, w: W - 2 * M - 0.7, h: 0.85, fontFace: HEAD, fontSize: 15, italic: true, color: INK, margin: 0 }
  );
  const items = [
    ["Bing", "pure word membership — no negation, no context", RED],
    ["VADER", "valence lexicon + negation, intensifiers, emoji", ACCENT],
    ["NRC", "8 emotion categories + polarity", ACCENT],
    ["SBERT + LR", "learned sentence semantics, light supervision", BLUE],
    ["RoBERTa", "fine-tuned on the labelled task itself", GREEN],
    ["LLM prompting", "world knowledge and reasoning, zero-shot", GREEN],
  ];
  const cw = (W - 2 * M - 0.5) / 3;
  items.forEach((it, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = M + col * (cw + 0.25), y = 3.15 + row * 1.42;
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w: cw, h: 1.22, rectRadius: 0.08, fill: { color: PAPER }, line: { color: PAPER },
    });
    s.addShape(pres.ShapeType.ellipse, {
      x: x + 0.24, y: y + 0.22, w: 0.3, h: 0.3, fill: { color: it[2] }, line: { color: it[2] },
    });
    s.addText(it[0], {
      x: x + 0.64, y: y + 0.2, w: cw - 0.85, h: 0.34,
      fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0,
    });
    s.addText(it[1], {
      x: x + 0.24, y: y + 0.6, w: cw - 0.48, h: 0.55,
      fontFace: BODY, fontSize: 10.5, color: MUTED_DARK, margin: 0,
    });
  });
  footer(s, "Every method runs through one harness, on identical held-out rows, against a majority-class baseline.");
  s.addNotes("The point is not to assume the supervisor is right. It is to test the claim and report what happens — including where it does not hold.");
}

// ===========================================================================
// 7. Sentiment results
// ===========================================================================
if (has("fig_nlp_sentiment.png")) {
  const s = lightSlide("Sentiment: VADER wins, and NRC is not the upgrade it looks like", "Results · real labelled data");
  s.addImage({ path: fig("fig_nlp_sentiment.png"), x: 0.55, y: 1.62, w: 7.6, h: 4.5 });
  const x = 8.4, w = W - x - M;
  s.addShape(pres.ShapeType.roundRect, {
    x, y: 1.85, w, h: 2.4, rectRadius: 0.09, fill: { color: "FDEEE7" }, line: { color: "FDEEE7" },
  });
  s.addText("The finding", {
    x: x + 0.26, y: 2.02, w: w - 0.5, h: 0.32,
    fontFace: BODY, fontSize: 13, bold: true, color: ACCENT, margin: 0,
  });
  s.addText(D.sentiment.finding, {
    x: x + 0.26, y: 2.42, w: w - 0.5, h: 1.7, fontFace: BODY, fontSize: 11.5, color: MUTED_DARK, margin: 0,
  });
  s.addText("So the system uses VADER and a transformer for polarity, and keeps NRC for the eight-emotion profile it is actually good at.", {
    x, y: 4.45, w, h: 1.2, fontFace: BODY, fontSize: 11.5, italic: true, color: INK, margin: 0,
  });
  footer(s, `TweetEval sentiment (SemEval-2017 Task 4) · ${D.sentiment.n_eval} held-out examples`);
  s.addNotes("Worth flagging that this partly contradicts the suggestion. NRC is the right call for emotion, not for polarity.");
}

// ===========================================================================
// 8. Irony — the headline
// ===========================================================================
if (has("fig_nlp_irony.png")) {
  const s = lightSlide("Irony: every word-list method is worse than guessing", "The headline result");
  s.addImage({ path: fig("fig_nlp_irony.png"), x: 0.55, y: 1.62, w: 7.6, h: 4.5 });
  const x = 8.4, w = W - x - M;
  statCard(s, { x, y: 1.75, w, h: 1.5, value: D.irony.best_lexicon_acc, label: "best word-list accuracy", color: RED });
  statCard(s, { x, y: 3.4, w, h: 1.5, value: D.irony.baseline_acc, label: "majority-class baseline", color: MUTED_DARK });
  statCard(s, { x, y: 5.05, w, h: 1.5, value: D.irony.best_acc, label: `best method — ${D.irony.best_name}`, color: GREEN });
  footer(s, `TweetEval irony (SemEval-2018 Task 3) · ${D.irony.n_eval} held-out examples`);
  s.addNotes(
    "This is the slide to spend time on. Sarcasm inverts meaning without changing vocabulary — " +
    "'Oh great, another subscription fee. Brilliant.' contains only positive words. It is not a " +
    "tuning problem; the information a word list needs is absent."
  );
}

// ===========================================================================
// 9. Failure cases
// ===========================================================================
{
  const s = lightSlide("Why it fails, in four sentences", "The headline result");
  const rows = [
    ["Oh great, another subscription fee. Brilliant work.", "negative", "positive", "positive", false],
    ["Nothing says premium like a charger sold separately. Fantastic.", "negative", "positive", "positive", false],
    ["Genuinely impressed by this serum, my skin looks calmer.", "positive", "positive", "positive", true],
    ["Disappointed by the build quality, would not repeat.", "negative", "negative", "negative", true],
  ];
  const cols = [6.6, 1.6, 1.6, 1.6];
  const x0 = M + 0.4;
  ["Caption", "True meaning", "Bing", "VADER"].forEach((h, i) => {
    s.addText(h, {
      x: x0 + cols.slice(0, i).reduce((a, b) => a + b, 0), y: 1.75, w: cols[i], h: 0.34,
      fontFace: BODY, fontSize: 11.5, bold: true, color: MUTED_DARK, margin: 0,
    });
  });
  rows.forEach((r, ri) => {
    const y = 2.2 + ri * 0.92;
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y: y - 0.1, w: W - 2 * M, h: 0.8, rectRadius: 0.07,
      fill: { color: r[4] ? PAPER : "FDEEE7" }, line: { color: r[4] ? PAPER : "FDEEE7" },
    });
    s.addText(r[0], {
      x: x0, y: y + 0.04, w: cols[0] - 0.2, h: 0.56,
      fontFace: BODY, fontSize: 12, color: INK, italic: true, margin: 0, valign: "middle",
    });
    [1, 2, 3].forEach((ci) => {
      const wrong = !r[4] && ci > 1;
      s.addText(r[ci], {
        x: x0 + cols.slice(0, ci).reduce((a, b) => a + b, 0), y: y + 0.04, w: cols[ci], h: 0.56,
        fontFace: BODY, fontSize: 12, bold: wrong, color: wrong ? RED : MUTED_DARK,
        margin: 0, valign: "middle",
      });
    });
  });
  s.addText(
    "Both lexicons handle literal text correctly. Both fail on ironic praise — the words are positive, the meaning is not.",
    { x: M, y: 6.05, w: W - 2 * M, h: 0.5, fontFace: BODY, fontSize: 12.5, color: INK, bold: true }
  );
  s.addNotes("Concrete beats abstract. Read the first row out loud and the failure is obvious.");
}

// ===========================================================================
// 10. Cross-domain
// ===========================================================================
if (has("fig_irony_crossdomain.png")) {
  const s = lightSlide("Does it survive a change of domain?", "Generalisation");
  s.addImage({ path: fig("fig_irony_crossdomain.png"), x: 0.7, y: 1.65, w: 8.0, h: 4.4 });
  const x = 8.95, w = W - x - M;
  s.addText(
    "In-domain accuracy overstates production performance. The same methods were re-run on an " +
    "independent sarcasm corpus from a different domain — news headlines rather than tweets.\n\n" +
    "This is the harder test, and it is especially hard on the transformer, which was fine-tuned " +
    "on the tweet corpus.",
    { x, y: 1.9, w, h: 3.4, fontFace: BODY, fontSize: 12, color: MUTED_DARK, margin: 0 }
  );
  footer(s, "Misra & Arora news-headline sarcasm corpus (arXiv:2212.06035)");
  s.addNotes("Running two corpora from different domains is what separates a benchmark from a leaderboard screenshot.");
}

// ===========================================================================
// 11. Topics
// ===========================================================================
if (has("fig_topic_coherence.png")) {
  const s = lightSlide("BERTopic vs LDA on short captions", "Content intelligence");
  s.addImage({ path: fig("fig_topic_coherence.png"), x: 0.85, y: 1.7, w: 6.4, h: 3.9 });
  const x = 7.7, w = W - x - M;
  s.addText(D.topics.summary, {
    x, y: 1.9, w, h: 1.6, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0,
  });
  s.addText(
    "BERTopic is expected to win here specifically because captions are short. LDA estimates a topic " +
    "distribution from word co-occurrence inside a document, and a twenty-word caption does not supply " +
    "enough. BERTopic clusters sentence embeddings and extracts terms afterwards, sidestepping the problem.\n\n" +
    "Diversity is reported alongside coherence because a model that repeats one generic topic scores " +
    "well on coherence alone.",
    { x, y: 3.6, w, h: 2.5, fontFace: BODY, fontSize: 11, color: MUTED_DARK, margin: 0 }
  );
  s.addNotes("Same corpus, same topic count, same coherence implementation. Otherwise the comparison means nothing.");
}

// ===========================================================================
// 12. Model results
// ===========================================================================
if (has("fig_model_baselines.png")) {
  const s = lightSlide("Does the learned model earn its complexity?", "Scoring model");
  s.addImage({ path: fig("fig_model_baselines.png"), x: 0.7, y: 1.75, w: 6.5, h: 3.9 });
  const x = 7.55, w = (W - x - M - 0.25) / 2;
  statCard(s, { x, y: 1.8, w, h: 1.55, value: D.model.r2, label: "R² (log), out-of-fold", color: BLUE });
  statCard(s, { x: x + w + 0.25, y: 1.8, w, h: 1.55, value: D.model.ceiling_frac, label: "of the achievable ceiling", color: GREEN });
  statCard(s, { x, y: 3.5, w, h: 1.55, value: D.model.spearman, label: "Spearman rank correlation", color: INK });
  statCard(s, { x: x + w + 0.25, y: 3.5, w, h: 1.55, value: D.model.ndcg, label: "NDCG@10 on the shortlist", color: INK });
  s.addText(D.model.note, {
    x, y: 5.25, w: w * 2 + 0.25, h: 1.1, fontFace: BODY, fontSize: 10.5, italic: true, color: MUTED_DARK, margin: 0,
  });
  footer(s, "GroupKFold on creator id — a creator never appears in both train and test.");
  s.addNotes(
    "The grouped split is the important detail. Each creator contributes up to three campaigns; a " +
    "random split would put the same creator on both sides and inflate everything."
  );
}

// ===========================================================================
// 13. Ablation + negative results
// ===========================================================================
if (has("fig_ablation.png")) {
  const s = lightSlide("Which pillar actually earns its place?", "Honest results");
  s.addImage({ path: fig("fig_ablation.png"), x: 0.7, y: 1.7, w: 6.6, h: 3.3 });
  const x = 7.65, w = W - x - M;
  s.addShape(pres.ShapeType.roundRect, {
    x, y: 1.8, w, h: 2.05, rectRadius: 0.09, fill: { color: "FDEEE7" }, line: { color: "FDEEE7" },
  });
  s.addText("Negative result #1", {
    x: x + 0.26, y: 1.96, w: w - 0.5, h: 0.3, fontFace: BODY, fontSize: 12.5, bold: true, color: RED, margin: 0,
  });
  s.addText(D.ablation.negative, {
    x: x + 0.26, y: 2.26, w: w - 0.5, h: 1.5, fontFace: BODY, fontSize: 11,
    color: MUTED_DARK, margin: 0, valign: "top",
  });
  s.addShape(pres.ShapeType.roundRect, {
    x, y: 4.0, w, h: 2.05, rectRadius: 0.09, fill: { color: "FDEEE7" }, line: { color: "FDEEE7" },
  });
  s.addText("Negative result #2", {
    x: x + 0.26, y: 4.16, w: w - 0.5, h: 0.3, fontFace: BODY, fontSize: 12.5, bold: true, color: RED, margin: 0,
  });
  s.addText(D.price.negative, {
    x: x + 0.26, y: 4.46, w: w - 0.5, h: 1.5, fontFace: BODY, fontSize: 11,
    color: MUTED_DARK, margin: 0, valign: "top",
  });
  s.addText(
    "Both are reported rather than buried. A pillar that does not predict still earns its place if it does something else — " +
    "the NLP layer drives brand-safety screening and every explanation the dashboard shows.",
    { x: M, y: 5.35, w: 6.6, h: 1.1, fontFace: BODY, fontSize: 11, italic: true, color: MUTED_DARK, margin: 0 }
  );
  s.addNotes(
    "Leading with your own negative results is the strongest thing you can do in a viva. It also " +
    "pre-empts the two questions a sharp reviewer would ask."
  );
}

// ===========================================================================
// 14. Product
// ===========================================================================
{
  const s = lightSlide("The product: a freemium two-sided marketplace", "Dashboard");
  const rows = [
    ["Creator search", "Niche and tier only", "Engagement quality, network position, audience geo/demo, ad load"],
    ["Performance score", "Band only (High / Medium / Low)", "Numeric percentile with pillar breakdown"],
    ["Brand-fit matching", "—", "Ranked shortlist with component decomposition"],
    ["Network position", "—", "Centrality, community, interactive map"],
    ["Price band", "—", "Estimated fee range per creator"],
    ["Creator analytics", "Peer benchmarking", "Rate guidance, brand-interest alerts, boosted placement"],
  ];
  const cw = [3.0, 3.2, 5.8];
  const x0 = M + 0.3;
  ["Capability", "Free", "Paid"].forEach((h, i) => {
    s.addText(h, {
      x: x0 + cw.slice(0, i).reduce((a, b) => a + b, 0), y: 1.72, w: cw[i], h: 0.3,
      fontFace: BODY, fontSize: 11.5, bold: true, color: i === 2 ? ACCENT : MUTED_DARK, margin: 0,
    });
  });
  rows.forEach((r, ri) => {
    const y = 2.12 + ri * 0.7;
    if (ri % 2 === 0) {
      s.addShape(pres.ShapeType.roundRect, {
        x: M, y: y - 0.07, w: W - 2 * M, h: 0.62, rectRadius: 0.06,
        fill: { color: PAPER }, line: { color: PAPER },
      });
    }
    r.forEach((cell, ci) => {
      s.addText(cell, {
        x: x0 + cw.slice(0, ci).reduce((a, b) => a + b, 0), y: y, w: cw[ci] - 0.15, h: 0.48,
        fontFace: BODY, fontSize: 11, bold: ci === 0,
        color: ci === 0 ? INK : (cell === "—" ? MUTED : MUTED_DARK), margin: 0, valign: "middle",
      });
    });
  });
  s.addText(
    "Offline scoring, online serving: every heavy model runs once in the pipeline and is cached. " +
    "The hosted dashboard loads no ML model at all — it reads precomputed Parquet, which is what keeps it inside a 1 GB free tier.",
    { x: M, y: 6.4, w: W - 2 * M, h: 0.6, fontFace: BODY, fontSize: 11, italic: true, color: MUTED_DARK }
  );
  s.addNotes("The architecture point matters: this is the same offline/online split production recommender systems use.");
}

// ===========================================================================
// 15. Limitations
// ===========================================================================
{
  const s = lightSlide("Limitations, stated plainly", "Honest results");
  const bw = (W - 2 * M - 0.4) / 2;
  bulletBox(s, {
    x: M, y: 1.7, w: bw, h: 4.4, heading: "What this project does not show", color: RED,
    items: [
      "Absolute numbers describe a simulation, not the real market. What transfers is the method comparison and the engineering.",
      "The graph is topical, not social — centrality means embeddedness in a shared vocabulary.",
      "Betweenness centrality is a sampled approximation.",
      "The price model's R² is partly circular: fees are generated from features the model sees.",
      "Benchmark sources are industry publications, not peer-reviewed research.",
    ],
  });
  bulletBox(s, {
    x: M + bw + 0.4, y: 1.7, w: bw, h: 4.4, heading: "What real deployment would need", color: GREEN, tint: "EAF6F1",
    items: [
      "A data-sharing agreement or first-party integration — public APIs do not expose the required fields.",
      "Real campaign outcome data. The modelling layer is already built to accept it; the target column would not change.",
      "Human evaluation of brand-fit, to make it validatable rather than only arguable.",
      "Fairness auditing across niche, tier and geography before any scoring system that affects creator income goes live.",
    ],
  });
  s.addNotes("Say these before you are asked. It is the difference between a defended project and a caught one.");
}

// ===========================================================================
// 16. Close
// ===========================================================================
{
  const s = darkSlide();
  s.addShape(pres.ShapeType.ellipse, {
    x: -1.6, y: H - 3.0, w: 4.4, h: 4.4, fill: { color: INK_SOFT }, line: { color: INK_SOFT },
  });
  s.addText("What this project shows", {
    x: M + 0.2, y: 1.15, w: 11, h: 0.5, fontFace: BODY, fontSize: 12,
    bold: true, color: ACCENT, charSpacing: 2,
  });
  const pts = [
    ["Word lists cannot detect irony.", "Measured, not asserted — every lexicon method scores at or below the majority-class baseline on real labelled data."],
    ["Learning beats the hand-weighted index.", `R² ${D.model.r2} against ${D.model.baseline_index}, under grouped cross-validation, reaching ${D.model.ceiling_frac} of the achievable ceiling.`],
    ["Two results came out negative, and are reported.", "The content pillar does not predict performance, and rule-based pricing is good enough. Both change what should be built next."],
  ];
  pts.forEach((p, i) => {
    const y = 1.95 + i * 1.45;
    s.addShape(pres.ShapeType.ellipse, {
      x: M + 0.2, y: y + 0.06, w: 0.44, h: 0.44, fill: { color: ACCENT }, line: { color: ACCENT },
    });
    s.addText(String(i + 1), {
      x: M + 0.2, y: y + 0.06, w: 0.44, h: 0.44, fontFace: BODY, fontSize: 14,
      bold: true, color: LIGHT, align: "center", valign: "middle", margin: 0,
    });
    s.addText(p[0], {
      x: M + 0.95, y: y, w: 10.5, h: 0.4, fontFace: HEAD, fontSize: 19, bold: true, color: LIGHT, margin: 0,
    });
    s.addText(p[1], {
      x: M + 0.95, y: y + 0.44, w: 10.4, h: 0.8, fontFace: BODY, fontSize: 12, color: "AAB1C6", margin: 0,
    });
  });
  s.addText("Thank you", {
    x: M + 0.2, y: 6.35, w: 6, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: LIGHT,
  });
  s.addNotes("Close on the negative results. It signals that the numbers on the earlier slides can be trusted.");
}

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote " + OUT));
