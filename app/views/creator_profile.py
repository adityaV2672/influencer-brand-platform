"""Creator OS — Profile. What a brand sees, and what the creator controls."""
from __future__ import annotations

import streamlit as st

from nectar import creator_ctx as ctx
from nectar import data, ui
from nectar.theme import (AMBER, AMBER_BG, GREEN, GREEN_BG, INK, INK_2, INK_3,
                          LINE, LINE_2)

me = ctx.me()
peers, peer_label = ctx.peers()

hdr, act = st.columns([3.3, 1])
with hdr:
    tick = (f"<span style='color:{GREEN};font-size:17px;margin-left:7px'>✓</span>"
            if bool(me.verified) else "")
    st.markdown(
        f"<div style='display:flex;align-items:flex-start;gap:16px'>"
        f"{ui.avatar(me.initials, me.avatar_color, 66)}"
        f"<div><div class='n-h1' style='font-size:27px'>{ui.esc(me.name)}{tick}</div>"
        f"<div class='n-sub'>{ui.esc(me.nectar_handle)} · {ui.esc(me.city)}</div>"
        f"<div style='font-size:13.5px;color:{INK_2};margin-top:6px'>{ui.esc(me.bio)}</div>"
        f"<div style='margin-top:10px'>"
        + "".join(ui.tag(c) for c in list(me.categories))
        + "".join(ui.tag(p) for p in list(me.platform_names))
        + "</div></div></div>", unsafe_allow_html=True)
with act:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.button("Edit profile", use_container_width=True, key="edit_profile")
    st.markdown(
        f"<div style='text-align:center;font-size:12px;color:{GREEN};margin-top:6px'>"
        f"{'✓ Verified' if me.verified else 'Not verified'}</div>",
        unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

plat = ui.platforms_of(me)
tiles = [(k, ui.count(v), "followers") for k, v in list(plat.items())[:2]]
tiles += [("Engagement", f"{me.engagement_rate * 100:.1f}%", "avg rate"),
          ("Growth", f"{me.follower_growth_rate * 100:+.1f}%", "per month")]
cols = st.columns(len(tiles))
for col, (lbl, val, sub) in zip(cols, tiles):
    with col:
        st.markdown(
            f"<div class='n-card' style='text-align:center;padding:14px'>"
            f"<div style='font-size:12.5px;color:{INK_3}'>{ui.esc(lbl)}</div>"
            f"<div class='n-num' style='font-size:23px;margin:2px 0'>{ui.esc(val)}</div>"
            f"<div style='font-size:11.5px;color:{INK_3}'>{ui.esc(sub)}</div></div>",
            unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

tabs = st.tabs(["Performance", "Audience", "Content", "Rates", "Availability"])


def row(label, value, benchmark=None, good=True):
    b = (f"<span style='font-size:12px;color:{INK_3};margin-right:14px'>"
         f"Benchmark: {benchmark}</span>" if benchmark else "")
    return (f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
            f"padding:13px 2px;border-bottom:1px solid {LINE_2}'>"
            f"<span style='font-size:13.5px'>{ui.esc(label)}</span>"
            f"<span>{b}<b class='n-num' style='color:{GREEN if good else AMBER}'>"
            f"{value}</b></span></div>")


with tabs[0]:
    pm = peers.median(numeric_only=True)
    html = [
        row("Avg engagement", f"{me.engagement_rate * 100:.1f}%",
            f"{pm.engagement_rate * 100:.1f}%", me.engagement_rate >= pm.engagement_rate),
        row("Avg reach / post", ui.count(me.avg_reach), ui.count(pm.avg_reach),
            me.avg_reach >= pm.avg_reach),
        row("Avg views / video", ui.count(me.avg_views), ui.count(pm.avg_views),
            me.avg_views >= pm.avg_views),
        row("Monthly growth", f"{me.follower_growth_rate * 100:+.1f}%",
            f"{pm.follower_growth_rate * 100:+.1f}%",
            me.follower_growth_rate >= pm.follower_growth_rate),
        row("Posts / month", f"{me.posting_frequency_month:.0f}",
            f"{pm.posting_frequency_month:.0f}",
            me.posting_frequency_month >= pm.posting_frequency_month),
    ]
    st.markdown("".join(html), unsafe_allow_html=True)
    st.markdown(f"<div class='n-muted' style='margin-top:10px'>"
                f"Benchmarks are the median of {len(peers):,} {ui.esc(peer_label)}.</div>",
                unsafe_allow_html=True)

with tabs[1]:
    # Audience authenticity first, because it is the question a brand asks
    # before any of the demographic detail matters. A creator whose following
    # is bought has an age distribution too.
    _aq = data.load("nectar_audience_quality.parquet")
    _row = (_aq[_aq.influencer_id.astype(str) == str(me.influencer_id)]
            if _aq is not None else None)
    if _row is not None and len(_row):
        _r = _row.iloc[0]
        _tone = {"Suspect": AMBER, "Mixed": AMBER}.get(str(_r.audience_band), GREEN)
        _bg = {"Suspect": AMBER_BG, "Mixed": AMBER_BG}.get(str(_r.audience_band), GREEN_BG)
        with st.container(border=True):
            _l, _m = st.columns([1, 2.4])
            with _l:
                st.markdown(
                    f"<div style='text-align:center;padding:8px 4px'>"
                    f"<div style='font-size:11.5px;color:{INK_3}'>AUDIENCE QUALITY</div>"
                    f"<div class='n-num' style='font-size:40px;color:{_tone};"
                    f"line-height:1.15'>{_r.audience_quality_score:.0f}</div>"
                    f"<div style='display:inline-block;padding:2px 9px;border-radius:6px;"
                    f"background:{_bg};color:{_tone};font-size:11.5px;font-weight:700'>"
                    f"{ui.esc(str(_r.audience_band))}</div></div>",
                    unsafe_allow_html=True)
            with _m:
                _cp = data.load("nectar_comment_profile.parquet")
                _c = (_cp[_cp.influencer_id.astype(str) == str(me.influencer_id)]
                      if _cp is not None else None)
                _rows = []
                if _c is not None and len(_c):
                    _cr = _c.iloc[0]
                    _rows = [
                        ("Automated-looking comments", f"{_cr.comment_automated_rate:.0%}"),
                        ("Repeated comment text", f"{_cr.comment_duplicate_rate:.0%}"),
                        ("Emoji-only comments", f"{_cr.comment_emoji_only_rate:.0%}"),
                        ("Average comment length",
                         f"{_cr.comment_mean_words:.1f} words"),
                    ]
                _rows.append(("Follower / following ratio",
                              f"{float(me.follower_following_ratio):.0f}×"))
                st.markdown("".join(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:7px 0;border-top:1px solid {LINE_2};font-size:12.5px'>"
                    f"<span style='color:{INK_2}'>{ui.esc(k)}</span>"
                    f"<span class='n-num'>{ui.esc(v)}</span></div>"
                    for k, v in _rows), unsafe_allow_html=True)
            st.markdown(
                f"<div class='n-muted' style='margin-top:10px;line-height:1.6'>"
                f"Scored from the comment section and the account's own signals. "
                f"On held-out creators the model reaches 0.89 macro F1; the "
                f"follower/following rule of thumb reaches 0.44. See the Metric "
                f"library for how every signal is produced.</div>",
                unsafe_allow_html=True)
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown("<div class='n-h3' style='margin-bottom:10px'>Age</div>",
                    unsafe_allow_html=True)
        for x in me.audience_age:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:8px'>"
                f"<span class='n-num' style='width:52px;font-size:12px'>{ui.esc(x['range'])}</span>"
                f"{ui.bar(x['pct'] / 100, INK, width='100%')}"
                f"<span class='n-num' style='width:36px;text-align:right;font-size:12px'>"
                f"{x['pct']}%</span></div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='n-h3' style='margin-bottom:10px'>Top locations</div>",
                    unsafe_allow_html=True)
        for x in me.audience_locations:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:8px'>"
                f"<span style='width:96px;font-size:12.5px'>{ui.esc(x['city'])}</span>"
                f"{ui.bar(x['pct'] / 100, GREEN, width='100%')}"
                f"<span class='n-num' style='width:36px;text-align:right;font-size:12px'>"
                f"{x['pct']}%</span></div>", unsafe_allow_html=True)

def _share_bar(shares: list[tuple[str, float, str]]) -> str:
    """A single stacked bar. Three separate progress bars made three unrelated
    numbers out of one distribution."""
    segs = "".join(
        f"<div style='width:{max(pct, 0) * 100:.4f}%;background:{colour}' "
        f"title='{ui.esc(label)} {pct:.0%}'></div>"
        for label, pct, colour in shares)
    key = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:5px;margin-right:14px'>"
        f"<span style='width:8px;height:8px;border-radius:2px;background:{colour}'></span>"
        f"<span style='font-size:11.5px;color:{INK_2}'>{ui.esc(label)} "
        f"<span class='n-num'>{pct:.0%}</span></span></span>"
        for label, pct, colour in shares)
    return (f"<div style='display:flex;height:9px;border-radius:5px;overflow:hidden;"
            f"background:{LINE_2};margin:8px 0 9px 0'>{segs}</div>"
            f"<div style='margin-bottom:4px'>{key}</div>")


def _tone_row(label: str, value: str, note: str = "") -> str:
    n = (f"<span style='font-size:11.5px;color:{INK_3};margin-left:8px'>"
         f"{ui.esc(note)}</span>" if note else "")
    return (f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
            f"padding:7px 0;border-top:1px solid {LINE_2};font-size:12.5px'>"
            f"<span style='color:{INK_2}'>{ui.esc(label)}{n}</span>"
            f"<span class='n-num'>{ui.esc(value)}</span></div>")


with tabs[2]:
    # ---- tone breakdown -------------------------------------------------
    # This block is what a brand looks at before it commits budget: not a
    # single safety number, but the distribution behind it. It was in the
    # first version of the dashboard, lost in the product rebuild, and is
    # back because content_safety is 12% of the fit composite and a brand
    # that cannot see what moved it cannot argue with it.
    tone_l, tone_r = st.columns(2, gap="large")

    with tone_l, st.container(border=True):
        st.markdown(ui.section("Caption tone",
                               "What the NLP pipeline read in the words."),
                    unsafe_allow_html=True)
        st.markdown(_share_bar([
            ("Positive", float(me.content_share_positive or 0), GREEN),
            ("Neutral", float(me.content_share_neutral or 0), INK_3),
            ("Negative", float(me.content_share_negative or 0), AMBER),
        ]), unsafe_allow_html=True)
        irony = float(me.content_irony_rate or 0)
        st.markdown(
            _tone_row("Irony / sarcasm rate", f"{irony:.0%}",
                      "CardiffNLP irony model")
            + _tone_row("Promotional posts", f"{float(me.content_promo_rate or 0):.0%}")
            + _tone_row("Disclosed partnerships",
                        f"{float(me.content_disclosure_rate or 0):.0%}")
            + _tone_row("Topic focus",
                        f"{int(me.content_n_topics or 0)} topics",
                        "lower is more predictable"),
            unsafe_allow_html=True)

    with tone_r, st.container(border=True):
        st.markdown(ui.section("Voice tone",
                               "What the delivery sounds like on video posts."),
                    unsafe_allow_html=True)
        card = (data.meta() or {}).get("audio_model", {})
        st.markdown(
            f"<div style='margin:-6px 0 8px 0'>"
            f"<span style='display:inline-block;padding:2px 8px;border-radius:5px;"
            f"background:{AMBER_BG};color:{AMBER};font-size:10.5px;font-weight:700;"
            f"letter-spacing:.04em'>VOICE ANALYSIS</span>"
            f"<span style='font-size:11.5px;color:{INK_3};margin-left:8px'>"
            f"trained late-fusion model over voice and caption</span></div>",
            unsafe_allow_html=True)
        st.markdown(_share_bar([
            ("Positive", float(getattr(me, "audio_share_positive", 0) or 0), GREEN),
            ("Neutral", float(getattr(me, "audio_share_neutral", 0) or 0), INK_3),
            ("Negative", float(getattr(me, "audio_share_negative", 0) or 0), AMBER),
        ]), unsafe_allow_html=True)
        mism = float(getattr(me, "tone_mismatch_rate", 0) or 0)
        st.markdown(
            _tone_row("Tone mismatch", f"{mism:.0%}",
                      "voice disagrees with the caption")
            + _tone_row("Speaking rate",
                        f"{float(getattr(me, 'audio_speech_rate_mean', 0) or 0):.0f} wpm")
            + _tone_row("Vocal energy",
                        f"{float(getattr(me, 'audio_arousal_mean', 0) or 0):.2f}",
                        "0 flat, 1 animated")
            + _tone_row("Spoken disclosure",
                        f"{float(getattr(me, 'spoken_disclosure_rate', 0) or 0):.0%}",
                        "says it out loud, not just in the caption")
            + _tone_row("Transcript confidence",
                        f"{float(getattr(me, 'asr_mean_confidence', 0) or 0):.2f}",
                        "ASR word confidence")
            + _tone_row("Video posts analysed",
                        f"{int(getattr(me, 'n_video_posts', 0) or 0)}"),
            unsafe_allow_html=True)
        if mism > 0.28:
            st.markdown(
                f"<div style='margin-top:10px;font-size:12px;color:{AMBER};"
                f"line-height:1.55'>Delivery contradicts the caption on more than "
                f"a quarter of video posts. Worth watching two before signing.</div>",
                unsafe_allow_html=True)

    st.markdown(
        f"<div class='n-muted' style='margin:14px 0 4px 0;line-height:1.6'>"
        f"Both panels feed one number. Content safety = 1 − 0.8×caption negative "
        f"− 0.5×irony − 0.30×voice negative − 0.20×tone mismatch, and is 12% of "
        f"the brand-fit composite. The two voice terms are weighted below the "
        f"caption terms, because the caption signal is the better measured "
        f"of the two.</div>",
        unsafe_allow_html=True)

    if card:
        with st.expander("How the voice label is produced, and what is real about it"):
            arms = {a["arm"]: a for a in card.get("arms", [])}
            st.markdown(
                f"**Architecture.** {card['architecture']}.\n\n"
                f"**Validation.** {card['validation']}, scored out of fold. Every "
                f"label shown above is the prediction made for that clip while "
                f"that creator was held out of training.")
            st.markdown(ui.table(
                ["Arm", "Accuracy", "Macro F1"],
                [[ui.esc(a["arm"]), f"{a['accuracy']:.4f}", f"{a['macro_f1']:.4f}"]
                 for a in card.get("arms", [])],
                aligns=["left", "right", "right"]), unsafe_allow_html=True)
            sweeps = card.get("sweeps", {})
            noise = sweeps.get("prosody_noise", [])
            curve = sweeps.get("learning_curve", [])
            wer = sweeps.get("wer", [])
            bullets = []
            if wer:
                bullets.append(
                    f"Raising the transcript's word error rate from "
                    f"{wer[0]['wer']:.0%} to {wer[-1]['wer']:.0%} costs the text "
                    f"branch only {wer[0]['text_macro_f1'] - wer[-1]['text_macro_f1']:.3f} "
                    f"macro F1 — the caption carries most of the text signal, so a "
                    f"cheaper recogniser would do.")
            if noise:
                bullets.append(
                    f"At ten times the recording noise the prosody head alone falls "
                    f"from {noise[0]['audio_macro_f1']:.3f} to "
                    f"{noise[-1]['audio_macro_f1']:.3f}, while fusion holds at "
                    f"{noise[-1]['fusion_macro_f1']:.3f}. Graceful degradation is what "
                    f"the second modality actually buys.")
            if curve:
                bullets.append(
                    f"The curve is still climbing at {curve[-1]['n_labelled_clips']:,} "
                    f"annotated clips ({curve[-1]['fusion_macro_f1']:.3f}), so a real "
                    f"build would need at least that many.")
            if bullets:
                st.markdown("**What the sweeps found.**\n\n"
                            + "\n\n".join(f"- {b}" for b in bullets))
            cav = card.get("caveats", {})
            st.markdown(
                f"**What is not real.** {cav.get('corpus', '')} "
                f"{cav.get('asr', '')} {cav.get('prosody_encoder', '')}\n\n"
                f"**What is real.** {cav.get('models', '')}\n\n"
                f"**Why fusion wins here.** {cav.get('why_fusion_wins', '')}")
            diag = card.get("corpus_diagnostics", {})
            if diag:
                st.markdown(f"**A defect this work found.** {diag['finding']} "
                            f"{diag['consequence']}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    posts = data.load("posts_sample.parquet")
    mine = posts[posts.influencer_id == me.influencer_id] if posts is not None else None
    st.markdown(
        f"<div class='n-muted' style='margin-bottom:12px'>"
        f"Recent posts with the labels each pipeline assigned. Caption sentiment is "
        f"RoBERTa and the irony probability is the CardiffNLP irony model; the voice "
        f"column is the voice track, and ⚠ marks the two disagreeing.</div>",
        unsafe_allow_html=True)
    if mine is None or mine.empty:
        st.markdown(ui.empty_state("✎", "No captions on file.",
                                   "Post-level NLP output is not available for this creator."),
                    unsafe_allow_html=True)
    else:
        # The voice track only exists for video posts, so a post with no audio
        # row shows a dash rather than being silently scored as neutral.
        au = data.load("nectar_audio_posts.parquet")
        au = (au.set_index("post_id") if au is not None
              else None)
        rows = []
        for p in mine.head(8).itertuples():
            irony = float(p.roberta_p_irony or 0)
            a = au.loc[p.post_id] if au is not None and p.post_id in au.index else None
            if a is None:
                voice = f"<span style='color:{INK_3};font-size:12.5px'>no video</span>"
            else:
                flag = (" <span title='voice disagrees with the caption'>⚠</span>"
                        if bool(a.tone_mismatch) else "")
                voice = (f"<span style='font-size:12.5px;color:"
                         f"{AMBER if bool(a.tone_mismatch) else INK_2}'>"
                         f"{ui.esc(str(a.audio_sentiment).title())}{flag}</span>")
            rows.append([
                f"<div style='font-size:13px;max-width:380px'>{ui.esc(p.caption)[:150]}</div>",
                ui.chip(str(p.roberta_sentiment).title()),
                voice,
                f"<span class='n-num' style='color:{AMBER if irony > 0.5 else INK_3}'>"
                f"{irony:.2f}</span>",
                f"<span style='font-size:12.5px;color:{INK_2}'>{ui.esc(p.topic_label)}</span>",
                f"<span class='n-num'>{ui.count(p.likes)}</span>",
            ])
        st.markdown(ui.table(
            ["Post", "Caption", "Voice", "P(irony)", "Topic", "Likes"], rows,
            aligns=["left", "left", "left", "right", "left", "right"]),
            unsafe_allow_html=True)

with tabs[3]:
    st.markdown(
        row("Reel", ui.inr(me.rate_reel)) +
        row("Story", ui.inr(me.rate_story)) +
        row("Carousel", ui.inr(me.rate_carousel)) +
        row("Suggested range",
            f"{ui.inr(me.price_low_inr)} – {ui.inr(me.price_high_inr)}"),
        unsafe_allow_html=True)
    st.markdown(
        f"<div class='n-muted' style='margin-top:12px'>These are the fee model's "
        f"predictions for a creator with your reach, engagement and niche — the same "
        f"numbers a brand sees. Story and Carousel are priced off the Reel rate at "
        f"0.35× and 0.70×.</div>", unsafe_allow_html=True)

with tabs[4]:
    fg, bg = ((GREEN, "#E8F4F0") if me.availability == "Available"
              else (AMBER, "#FBF3E0"))
    st.markdown(
        f"<div style='background:{bg};border-radius:12px;padding:16px 18px'>"
        f"<div style='font-size:12.5px;color:{INK_2}'>Current status</div>"
        f"<div style='font-size:20px;font-weight:700;color:{fg}'>"
        f"{ui.esc(me.availability)}</div>"
        f"<div style='font-size:13px;color:{INK_2};margin-top:4px'>"
        f"{ui.esc(me.available_window)}</div></div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div class='n-muted' style='margin-top:14px'>Availability is a hard filter "
        f"on the brand side. A creator marked unavailable is excluded from search "
        f"results regardless of how well they score.</div>", unsafe_allow_html=True)
