// desktop.yml's resolve step (spec §9): which commit to build, with which release identity, and whether
// to publish. Node 24 loads tags.ts as it is, so a tag passes exactly the rules the app's build applies.
import { execFileSync } from 'node:child_process';
import { appendFileSync } from 'node:fs';
import { parseReleaseTag } from '../src/release/tags.ts';

const env = process.env;
const git = (...args) => execFileSync('git', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
function fail(message) { console.log(`::error::${message}`); process.exit(1); }

const dryRun = env.DRY_RUN === 'true';
const tag = (env.GITHUB_EVENT_NAME === 'push' ? env.GITHUB_REF_NAME : env.INPUT_TAG) ?? '';
const identityTag = env.INPUT_IDENTITY ?? '';
for (const t of [tag, identityTag]) if (t && !parseReleaseTag(t)) fail(`${JSON.stringify(t)} is not a release tag (RELEASING.md)`);
if (tag && identityTag) fail('Give either tag or identity_tag, not both');
if (identityTag && !dryRun) fail('identity_tag is for dry runs only');
if (!tag && !identityTag && !dryRun) fail('A build without a tag must be a dry run');

// Resolved once: both build jobs check out this commit, even if the tag moves during the run.
let sha = env.GITHUB_SHA;
if (tag) {
  let type = '';
  try { type = git('cat-file', '-t', `refs/tags/${tag}`); } catch { /* no such tag */ }
  if (type !== 'tag') fail(`${tag} must be an existing annotated tag`);
  try { sha = git('rev-parse', `refs/tags/${tag}^{commit}`); }
  catch { fail(`${tag} must point to a commit`); }
}
let build = true;
try { git('cat-file', '-e', `${sha}:desktop/src/release/tags.ts`); }
catch { build = false; console.log(`::notice::${tag || sha} has no M4 desktop code; nothing to build`); }
const outputs = { tag, identity: tag || identityTag, sha, build, publish: build && Boolean(tag) && !dryRun };
appendFileSync(env.GITHUB_OUTPUT, Object.entries(outputs).map(([key, value]) => `${key}=${value}\n`).join(''));
