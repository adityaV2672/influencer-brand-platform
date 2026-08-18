# Start here

Plain-language guide to your project. No prior technical knowledge assumed.
Everything is already installed on your machine — this explains what exists,
how to run it, and what to do if something breaks.

---

## 1. What this project is, in one paragraph

A website where **brands** search for **social-media creators** to advertise their
products. Instead of ranking creators by follower count (which is what the industry
actually does, and which is misleading), it predicts how well each creator will
*perform* in a paid campaign, using a model trained on past campaign results. It
also estimates what each creator should cost, checks whether a creator has recently
promoted a competitor, and analyses what each creator talks about and in what tone.

---

## 2. Where everything lives

Everything is in one folder: `C:\Users\adity\influencer-platform`

| Folder | What's inside | Do you touch it? |
|---|---|---|
| `app\` | The dashboard (the website) | No |
| `src\` | All the code that does the analysis | No |
| `data\` | Raw and processed data | No |
| `artifacts\` | Everything the analysis produced | No |
| `app_data\` | The small files the live website reads | No |
| `reports\` | **Your report, slides and charts** | **Yes — this is your submission** |

Two files you'll double-click:

- **`run_dashboard.bat`** — opens the dashboard in your browser
- **`rebuild_all.bat`** — re-runs the whole analysis from scratch (~2.5 hours)

---

## 3. The three things you're submitting

| Deliverable | Where it appears |
|---|---|
| Written report | `reports\Influencer_Platform_Report.docx` |
| Slide deck | `reports\Influencer_Platform_Deck.pptx` |
| Live dashboard | a public web link (see `DEPLOY.md`) |
| Code | your GitHub repository (see `DEPLOY.md`) |

---

## 4. How to open the dashboard

Double-click **`run_dashboard.bat`**.

A black window opens (that's normal — leave it open) and your browser opens at
`http://localhost:8501`.

**To stop it:** close the black window, or click in it and press `Ctrl + C`.

### What you'll see

There's a **Plan** switch in the left sidebar — **Free** and **Paid**. Flip it back
and forth. That is the freemium business model from your proposal, working: on Free
you get bands instead of numbers, no brand matching, no network view, no prices.
It's worth demonstrating this live if you present.

Five pages:

1. **Discover** — the brand-side search. Filter creators, see them ranked.
2. **Creator profile** — one creator in depth: score breakdown, tone, network, posts.
3. **Model & methods** — **this is the page for your professor.** It has the NLP
   method comparison, the model results, the ablation, and an explicit page
   section on what is real data and what is synthetic.
4. **Network map** — the creator graph, interactive.
5. **Creator analytics** — the creator side of the marketplace.

---

## 5. If you need to re-run the analysis

Only necessary if you change something. Double-click **`rebuild_all.bat`**.

It runs 13 steps and takes roughly 2.5 hours. You can watch progress in
`pipeline_log.txt`, which updates live.

**It is safe to interrupt.** The slowest step saves its progress every 1,000 posts,
so restarting picks up where it left off rather than starting over.

To run just one step:

```powershell
cd $env:USERPROFILE\influencer-platform
$env:PYTHONPATH = $PWD
& "$env:USERPROFILE\anaconda3\envs\influencer\python.exe" run_pipeline.py --list
& "$env:USERPROFILE\anaconda3\envs\influencer\python.exe" run_pipeline.py --only report
```

---

## 6. The four things to say if you're questioned

Your professor's feedback was that the qualitative analysis was weak and that Bing
sentiment is primitive. Here's how the project answers that — and where it
respectfully disagrees.

**1. "Word lists can't detect sarcasm" — proved, not asserted.**
Every method was run on real, human-labelled data (TweetEval and the Misra & Arora
sarcasm corpus — real tweets and headlines annotated by people). On the irony task,
the lexicon methods score at or below the majority-class baseline. Concretely: both
Bing and VADER score *"Oh great, another subscription fee. Brilliant work."* as
**positive**. The words are positive; the meaning is not.

**2. NRC is not a straight upgrade — and the data says so.**
NRC scores *worse* than VADER on three-class polarity. Its strength is its eight
emotion categories, not positive/negative. So the project uses NRC for the emotion
profile and something else for polarity. If asked "did you just do what you were
told?", the answer is no — it was tested, and the result shaped the design.

**3. Two structural problems in the original proposal were found and fixed.**
The network-analysis section assumed access to a follower graph. Instagram does not
give that to third parties, at all — so that pillar was rebuilt on data that
genuinely exists (shared hashtags and shared brand collaborations). And the original
design had no target variable to train on, which meant "Phase 1" wasn't machine
learning at all — it was a scoring rubric. That's fixed too.

**4. Two results came out negative, and they're in the report.**
The content/NLP features don't improve campaign prediction, and the machine-learned
price model barely beats a simple rate card. Both are reported rather than hidden.
Leading with your own negative results is the single strongest thing you can do in a
viva — it signals that the positive numbers can be trusted.

---

## 7. The honest limitation, stated up front

**The 2,000 creators are synthetic.** They're generated, not scraped. This is worth
saying before you're asked, along with the reason: there is no legal way to obtain
per-post engagement data for arbitrary Instagram creators, and building the project
on scraped data would have been both against platform terms and unreproducible.

What makes it defensible rather than a cop-out:

- The synthetic data is **calibrated to published industry benchmarks** — every
  follower tier's engagement rate and fee band falls inside the published range,
  and that's verified by an automated test, not by eye.
- **The NLP results use real data.** This matters, because synthetic text cannot
  validate an NLP method: if you generate a caption, label it "sarcastic", and then
  measure whether a detector finds it, you've measured whether the detector can
  reverse-engineer your template — not whether it detects sarcasm.
- Because the noise level in the simulation is known, the **maximum achievable
  accuracy is computable** — so the model's score is reported as a fraction of what
  is achievable, not as a bare number that could mean anything.

---

## 8. If something breaks

**The dashboard says "Creator database is not available"**
The analysis hasn't finished, or `app_data\` is empty. Run `rebuild_all.bat`.

**The black window closes instantly**
The conda environment is missing. Re-run `setup_env.ps1`:
```powershell
powershell -ExecutionPolicy Bypass -File $env:USERPROFILE\influencer-platform\setup_env.ps1
```

**`rebuild_all.bat` seems frozen**
Check `pipeline_log.txt` — if its timestamp is updating, it's working. The
transformer step genuinely takes about 80 minutes on this machine.

**Everything looks broken and you want to start clean**
Delete the `artifacts` and `app_data` folders, then run `rebuild_all.bat`. Nothing
in `src\` or `app\` is ever modified by a run, so the code can't be corrupted by one.

**Checking the results are sound**
```powershell
cd $env:USERPROFILE\influencer-platform
$env:PYTHONPATH = $PWD
& "$env:USERPROFILE\anaconda3\envs\influencer\python.exe" -m tests.test_integrity
```
This verifies calibration against published benchmarks, checks for data leakage,
confirms the cross-validation is genuinely grouped, and re-checks the report's
headline claim against the actual numbers. It should print `0 FAILED`.

---

## 9. Deadline checklist

- [ ] Analysis has run end to end (`pipeline_log.txt` ends with all stages ok)
- [ ] `python -m tests.test_integrity` prints `0 FAILED`
- [ ] Dashboard opens and all five pages load
- [ ] Free/Paid switch visibly changes what's shown
- [ ] Report opens and has no `[figure missing]` placeholders
- [ ] Slide deck opens
- [ ] Code pushed to GitHub (`DEPLOY.md` step 2)
- [ ] Dashboard deployed and the public link works from your phone (`DEPLOY.md` step 3)
- [ ] Public link pasted into the report and the final slide
