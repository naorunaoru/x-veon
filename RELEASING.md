# Releasing

X-veon deploys to GitHub Pages as two channels on one site:

| Channel | URL | What it is | Source of truth |
|---|---|---|---|
| stable | https://naorunaoru.github.io/x-veon/ | What everyone uses. Never rebuilt at deploy time. | The GitHub Release marked **Latest** (its tag must start with `stable/`). Its `site-stable.zip` asset *is* the site. |
| beta | https://naorunaoru.github.io/x-veon/beta/ | Current `develop`, for early users. | The newest **annotated** `beta/*` tag (by tagger date), built from source. |

Nothing deploys on a branch push. A deploy happens on a `beta/*` tag push, on promotion, or on a manual dispatch of `deploy.yml`, which simply re-assembles the current channels. Every deploy rebuilds the whole site (stable from its bundle, beta from source), because Pages serves one artifact per repo.

Both workflow files must be identical on `develop` and `main`: a `beta/*` tag push runs the copy at the tag (develop's), while `gh workflow run … --ref main` and the promote chain run `main`'s copy. After changing a workflow on `develop`, mirror the same change to `main` in its own commit.

The channels keep **separate libraries** in the browser (separate IndexedDB / OPFS namespaces). Switching channels shows an empty library.

Tags are `beta/YYYY-MM-DD` and `stable/YYYY-MM-DD`; a second one on the same day appends `-2`, `-3`, … **Always annotated** (`git tag -a`). A lightweight `beta/*` tag fails its own deploy run and is otherwise ignored.

## Ship a beta

```bash
git checkout develop && git pull
git tag -a beta/$(date -u +%F) -m "beta: <what changed>"
git push origin beta/$(date -u +%F)
```

If today's name is taken, use `beta/$(date -u +%F)-2`, and so on. Watch it with `gh run watch`.

## Roll beta back

Create a new annotated tag that points at the older commit; being newest by date, it wins:

```bash
git tag -a beta/$(date -u +%F)-2 -m "rollback to beta/<old>" 'beta/<old>^{}'
git push origin beta/$(date -u +%F)-2
```

(`^{}` peels the old tag to its commit.)

## Promote beta to stable

```bash
git checkout main && git pull
git merge --no-ff develop && git push        # nothing deploys on this push
gh workflow run promote.yml --ref main -f ref=main
```

This builds `main` as stable, creates `stable/<date>` plus a release carrying the bundle, marks it Latest, and redeploys both channels.

## Roll stable back

```bash
gh release list
gh release edit stable/<old> --latest
gh workflow run deploy.yml --ref main
```

## Dry runs

Both workflows can run without publishing anything, to check a workflow or layout change on a branch before it ships:

```bash
gh workflow run deploy.yml --ref <branch> -f dry_run=true -f beta_ref=<branch>   # builds beta from <branch>; no upload, no deploy
gh workflow run promote.yml --ref <branch> -f ref=<branch> -f dry_run=true       # builds stable from <branch>; no tag, release or deploy
```

The build steps call `scripts/build-web.sh <channel>` when the checkout has it, and fall back to the older `web/`-only steps otherwise, so `main`'s copy of these files builds either layout.

## What is live right now

```bash
gh release view --json tagName --jq .tagName
git fetch --tags && git for-each-ref 'refs/tags/beta/*' --sort=-taggerdate --format='%(refname:short)  %(taggerdate:short)' | head -3
gh run list --workflow=deploy.yml --limit 5
```

## Bump the RAW decoder

The decoder is the git submodule `shared/crates/vendor/rawloader` (the user's rawloader fork). To move it: push the new commit to the fork, then in this repo run `git -C shared/crates/vendor/rawloader fetch origin && git -C shared/crates/vendor/rawloader checkout <sha>`, rebuild with `npm run build:wasm:decoder --workspace shared`, test with a RAF and an ARW, and commit the updated gitlink on `develop`. The next beta tag picks it up; CI needs nothing else because checkouts use `submodules: true`.

## Local builds

`npm run build` and `npm run preview` need `XV_CHANNEL` (`stable`, `beta` or `dev`); the dev server defaults to `dev`. Run them from the repository root.

`npm run dev` and `npm run preview` both serve without cross-origin isolation, like GitHub Pages (which sends no COOP/COEP headers).

```bash
XV_CHANNEL=beta npm run build && XV_CHANNEL=beta npm run preview   # http://localhost:4173/x-veon/beta/
```

`scripts/build-web.sh <stable|beta|dev>` is the exact recipe CI runs: a clean install of the `shared` and `web` workspaces, the tests, the wasm builds, the lens data, the site build and the typecheck, then the output checks. It leaves the site in `web/dist`.
