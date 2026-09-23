# Setup — from zero to both of you committing

Written for someone who has never used git. Every command says what it does and what you should see.
Serena does **Part 2**; PK skips it. Everything else you both do.

---

## Part 0 — What git actually is, in four sentences

Git keeps a history of your project. GitHub is a website that stores a copy of that history so two people
can share it. You work on your own computer, then **push** your changes up and **pull** theirs down. The
whole discipline is: pull before you start, push when you finish, and never both edit the same file at the
same time.

---

## Part 1 — Install things (both of you, once)

**1.1 — GitHub account.** [github.com](https://github.com) → Sign up. Use a name PK will recognise.

**1.2 — Git.**

- **Mac:** open Terminal and type `git --version`. If it offers to install developer tools, say yes. Done.
- **Windows:** download from [git-scm.com](https://git-scm.com/download/win), run it, click Next on
  everything. This gives you a program called **Git Bash** — use that, not Command Prompt, so every command
  below works the same as on a Mac.

Check it worked:

```bash
git --version
```

You should see something like `git version 2.43.0`.

**1.3 — Tell git who you are.** This is what gets stamped on your commits.

```bash
git config --global user.name "Serena Sun"
git config --global user.email "sunkaiwen1023@gmail.com"
```

Use the same email as your GitHub account or your commits won't link to your profile.

**1.4 — Python.** You need 3.10 or newer.

```bash
python3 --version
```

If that fails or shows 3.9 or lower, install from [python.org](https://python.org).

**1.5 — Optional but recommended: GitHub Desktop.** [desktop.github.com](https://desktop.github.com). It's
a point-and-click version of everything below. You can use it for the everyday stuff and drop to the
terminal when something goes wrong. If one of you prefers clicking and the other prefers typing, that's
completely fine — you're both talking to the same repo.

---

## Part 2 — Create the repo (Serena only)

**2.1** Go to [github.com/new](https://github.com/new).

- **Repository name:** `sketch2stl`
- **Description:** `Hand-drawn 2D sketches to 3D-printable STL. CMU 24-679 Project 1.`
- **Private** — turn this on for now. You can flip it public before the presentation.
- **Do NOT tick** "Add a README", "Add .gitignore" or "Choose a license". The folder I gave you already has
  those, and ticking them creates a conflict on your very first push.

Click **Create repository**. Leave the page open — you'll need the URL.

**2.2 — Invite PK.** Settings → Collaborators → **Add people** → his GitHub username → **Add**. He gets an
email; he must accept it before he can push.

**2.3 — Upload the starter folder.** Unzip `sketch2stl.zip` somewhere sensible — `~/Documents/sketch2stl`,
not Downloads, and **not inside a folder that iCloud or OneDrive syncs**. Cloud sync and git fight each
other and you will lose work. Then:

```bash
cd ~/Documents/sketch2stl
git init
git add .
git commit -m "Initial scaffold: geometry kernel, recogniser stubs, Gradio UI, tests"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/sketch2stl.git
git push -u origin main
```

Replace `YOUR-USERNAME`. Git will ask you to sign in — a browser window opens, approve it.

Refresh the GitHub page. Your files are there.

**2.4 — Protect `main`.** This is the single highest-value five minutes in this whole guide.

Settings → Branches → **Add branch ruleset** → name it `protect-main` → Enforcement status **Active** →
under *Target branches* add `main` → tick **Require a pull request before merging**. Set required approvals
to **0** (with two people, waiting for approval just blocks you).

What this buys you: neither of you can accidentally push broken code straight to `main`. Every change goes
through a pull request, which means you both see it.

---

## Part 3 — Get the code onto your computer (both of you)

PK does this after accepting the invite. Serena already has the folder, so she skips to Part 4.

```bash
cd ~/Documents
git clone https://github.com/SERENAS-USERNAME/sketch2stl.git
cd sketch2stl
```

You now have the whole project and its history.

---

## Part 4 — Python environment (both of you)

A **virtual environment** is a private box of libraries for this project, so installing something here
can't break anything else on your computer.

```bash
cd ~/Documents/sketch2stl
python3 -m venv .venv
```

Turn it on — **you do this every time you open a new terminal**:

```bash
source .venv/bin/activate          # Mac / Linux / Git Bash
.venv\Scripts\activate             # Windows Command Prompt
```

Your prompt now starts with `(.venv)`. That's how you know it's on.

Install everything:

```bash
pip install -r requirements.txt
```

Takes a few minutes. Then check it worked:

```bash
pytest -q
```

You should see **31 passed, 1 xfailed**. The `xfailed` is deliberate — it's the rectangle-recognition test,
written before the feature exists so it flips to passing the moment Serena implements it.

Now run the app:

```bash
python app.py
```

Open the link it prints. Click **Add volume**, then **Cut volume**, then **Export STL**. You get the plate
from the mockup. That's your starting point — every piece of it is real code you'll replace.

---

## Part 5 — The daily loop

This is the whole workflow. Learn these six commands and you're done.

**Before you start work, every single time:**

```bash
git checkout main
git pull
```

*"Switch to the main branch, and download anything PK has added."* Skipping this is the cause of roughly
every merge conflict you will ever have.

**Make a branch for what you're about to do:**

```bash
git checkout -b serena/rect-recognition
```

*"Make a new branch and switch to it."* Name it `yourname/what-it-does`. A branch is a private workspace —
you can break things freely without affecting PK.

**Work. Then save your progress:**

```bash
git add .
git commit -m "Add rectangle fitting to RuleRecognizer"
```

*`add` picks what to save, `commit` saves it with a note.* Commit often — every time something works, not
once at the end of the day. Small commits are much easier to undo.

**Send it to GitHub:**

```bash
git push -u origin serena/rect-recognition
```

The `-u origin branch-name` part is only needed the first time on a new branch. After that just `git push`.

**Open a pull request.** Go to GitHub — there's a green "Compare & pull request" button. Click it, write a
sentence about what changed, click **Create pull request**. Message PK. When he's had a look, click **Merge
pull request**, then **Delete branch**.

**Then go back to the start:**

```bash
git checkout main
git pull
```

---

## Part 6 — The four rules that prevent 90% of the pain

**1. Own whole files, never parts of files.** The ownership table in `CONTRIBUTING.md` says who owns what.
Git merges two people editing *different files* perfectly and automatically. It struggles when you both
edit the *same lines*. Staying in your own files means you will almost never see a conflict.

**2. Never commit directly to `main`.** Always a branch, always a pull request. Part 2.4 enforces this so
you can't forget.

**3. Never commit large files.** No STLs, no datasets, no `.venv`, no model weights. The `.gitignore` blocks
the common ones. GitHub rejects anything over 100 MB and cleaning it out of history afterwards is genuinely
horrible.

**4. `sketch2stl/types.py` and `config.py` are shared.** Changing them changes the other person's code
underneath them. Message each other first, make the change in its own small pull request, merge it fast,
and both `git pull` immediately.

---

## Part 7 — When it goes wrong

**"Updates were rejected because the remote contains work that you do not have"**

PK pushed while you were working. Fix:

```bash
git pull --rebase
git push
```

*"Take my commits off, get his, put mine back on top."*

---

**MERGE CONFLICT — `CONFLICT (content): Merge conflict in <file>`**

You both changed the same lines. Not dangerous, just fiddly. Open the file and find:

```
<<<<<<< HEAD
        depth = 10.0
=======
        depth = 12.0
>>>>>>> pk/depth-default
```

The top block is yours, the bottom is his. Delete the three marker lines and whichever version is wrong,
leaving only the code you want. Then:

```bash
git add .
git rebase --continue      # if you were rebasing
git commit                 # if you were merging
```

If you get lost, `git rebase --abort` puts everything back exactly as it was. Nothing is lost.

---

**"I committed to `main` by accident"**

```bash
git branch serena/my-work        # bookmark the work
git reset --hard origin/main     # put main back how it was
git checkout serena/my-work      # carry on on the branch
```

---

**"I committed a huge file"**

If you haven't pushed yet:

```bash
git reset --soft HEAD~1
git restore --staged path/to/the-big-file
```

Then add it to `.gitignore` and commit again. If you already pushed, stop and ask for help before doing
anything else — the fix rewrites history and you should do it together.

---

**"I've broken everything and just want PK's version back"**

Your uncommitted changes are thrown away by this, so copy anything you want to keep somewhere else first:

```bash
git fetch origin
git reset --hard origin/main
```

---

## Part 8 — Deploying to Hugging Face Spaces

Do this in week 3, not before — but know it's a two-minute job, not a project.

1. [huggingface.co/new-space](https://huggingface.co/new-space) → name `sketch2stl` → SDK **Gradio** → CPU
   basic (free).
2. A Space is itself a git repo, so you push to it like any other remote:

```bash
git remote add space https://huggingface.co/spaces/sunkaiwen/sketch2stl
git push space main
```

3. It builds from `requirements.txt` and runs `app.py`. Watch the build log for missing packages.

One thing to check first: `trimesh`'s boolean operations need `manifold3d`, which is in `requirements.txt`
but occasionally fails to build on a Space. If it does, the log will say so — that's a known issue with a
known fix, not a mystery, so budget an hour for it rather than discovering it the night before.

---

## First hour, together

Sit down with PK for an hour and do this in order. It front-loads every decision that would otherwise cause
a painful merge later.

1. Both of you get through Part 4 and see **31 passed**.
2. Both run `python app.py` and export an STL. Open it in a slicer or a 3D viewer so you've both seen the
   output the project is aiming at.
3. Read `docs/data_contract.md` out loud, together. It's short. Agree on it or change it now — that file
   is what lets you work separately.
4. Read the ownership table in `CONTRIBUTING.md`. Change it if the split doesn't match what you each want
   to build.
5. Open `docs/decisions.md` and fill in the three open decisions at the bottom. Put dates on them.
6. Each of you make a branch, change one comment, push, open a PR, merge it. Do the whole loop once while
   the other is watching. Getting the first PR out of the way removes most of the fear.
