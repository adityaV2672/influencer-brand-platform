# Deploying the dashboard — step by step

You need a free GitHub account and a free Streamlit account. Nothing is paid at
any point. Total time: about 10 minutes.

If a step doesn't look like what's described, stop and say what you see rather
than guessing — every step here is reversible except deleting things.

---

## Step 1 — Create an empty repository on GitHub

1. Go to **https://github.com/new** (sign in if asked).
2. **Repository name:** `influencer-brand-platform`
3. **Description:** `ML platform scoring creators on predicted campaign performance`
4. Choose **Public**.
   *It must be Public — Streamlit's free tier can only deploy public repos.*
5. **Do NOT tick** "Add a README file", "Add .gitignore", or "Choose a license".
   The project already has these, and ticking them causes a conflict on the first push.
6. Click **Create repository**.

You'll land on a page showing setup commands. Ignore them — use the ones below instead.

---

## Step 2 — Push the project

Open **PowerShell** (press `Win`, type `powershell`, press Enter) and paste this
**one line at a time**, pressing Enter after each. Replace `YOUR-USERNAME` with
your actual GitHub username.

```powershell
cd $env:USERPROFILE\influencer-platform
```

```powershell
& "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe" init -b main
```

Tell git who you are (use the email on your GitHub account):

```powershell
& "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe" config user.name "Aditya Verma"
```

```powershell
& "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe" config user.email "adityavrm26@gmail.com"
```

Stage and commit everything:

```powershell
& "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe" add -A
```

```powershell
& "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe" commit -m "Influencer-brand collaboration platform"
```

Connect to GitHub and push:

```powershell
& "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe" remote add origin https://github.com/YOUR-USERNAME/influencer-brand-platform.git
```

```powershell
& "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe" push -u origin main
```

**A browser window or a popup will appear asking you to sign in to GitHub.**
Sign in and approve. This happens once; git remembers afterwards.

> **If instead you see a prompt asking for a *password* in the terminal:** GitHub
> no longer accepts account passwords there. Close it, go to
> https://github.com/settings/tokens → *Generate new token (classic)* → tick the
> **repo** checkbox → Generate → copy the token, and paste that as the password.
> Your username stays the same.

When it finishes, refresh your GitHub repo page. You should see all the files.

---

## Step 3 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io**
2. Click **Sign in with GitHub** and authorise it.
3. Click **Create app**, then **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `YOUR-USERNAME/influencer-brand-platform`
   - **Branch:** `main`
   - **Main file path:** `app/Home.py`   ← this one matters, don't leave it as `streamlit_app.py`
5. Click **Deploy**.

The first build takes 3–6 minutes while it installs the packages. You'll see a
log scrolling. When it finishes you get a public URL like:

```
https://influencer-brand-platform.streamlit.app
```

That link is what goes in your report and your slides. Anyone can open it —
no login needed.

---

## Why this is set up the way it is

The dashboard loads **no machine-learning models at all**. Every heavy
computation — embeddings, transformers, topic modelling, model training — runs
once on your machine in the pipeline and is saved to `app_data/` as small
Parquet files, which are committed to the repo on purpose.

That's why there are two requirements files:

| File | Used by | Contains |
|---|---|---|
| `requirements.txt` | Streamlit Cloud | streamlit, pandas, plotly, networkx — nothing heavy |
| `requirements-dev.txt` | Your machine | the above **plus** torch, transformers, BERTopic, LightGBM |

Streamlit's free tier gives roughly 1 GB of memory. Installing PyTorch alone
would exceed it. Keeping the two separate is what makes the free tier work — and
it's the same offline-scoring / online-serving split that production
recommendation systems use, so it's a design decision worth mentioning if you're
asked about architecture.

---

## Updating the app later

Any time you re-run the pipeline and want the live site to reflect it:

```powershell
cd $env:USERPROFILE\influencer-platform
& "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe" add -A
& "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe" commit -m "Update results"
& "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe" push
```

Streamlit redeploys automatically within a minute or two.

---

## Troubleshooting

**"Repository not found" on push** — the repo name or username in the `remote add`
line is wrong. Check it with:
```powershell
& "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe" remote -v
```
Fix it with `remote set-url origin https://github.com/CORRECT-NAME/influencer-brand-platform.git`

**"Updates were rejected"** — you ticked one of the "Add a README" boxes in Step 1.
Run this once, then push again:
```powershell
& "$env:USERPROFILE\anaconda3\envs\influencer\Library\bin\git.exe" pull --rebase origin main
```

**App deploys but shows "Creator database is not available"** — `app_data/` didn't
get committed. Check that `app_data/influencers.parquet` exists locally and that
`.gitignore` does not list `app_data`. It is excluded from the ignore list on purpose.

**App crashes with "Error installing requirements"** — Streamlit picked up
`requirements-dev.txt`. In your app's settings on share.streamlit.io, confirm the
main file path is `app/Home.py`; Streamlit reads `requirements.txt` from the repo
root automatically.

**Running it locally instead** — double-click `run_dashboard.bat` in the project
folder, or:
```powershell
& "$env:USERPROFILE\anaconda3\envs\influencer\python.exe" -m streamlit run app\Home.py
```
