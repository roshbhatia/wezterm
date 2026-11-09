#!/usr/bin/env bash
set -euo pipefail

upstream_url="${1:?upstream Git URL required}"
check_workflow="${2:?validation workflow required}"
base=main
branch=automation/upstream-sync
repository="${GITHUB_REPOSITORY:?repository required}"

git fetch --no-tags origin "refs/heads/${base}:refs/remotes/origin/${base}"
git fetch --no-tags "${upstream_url}" "refs/heads/main:refs/remotes/upstream/main"
if git merge-base --is-ancestor refs/remotes/upstream/main "origin/${base}"; then
  echo 'Upstream is already included in main.'
  exit 0
fi

previous=""
if git ls-remote --exit-code --heads origin "${branch}" > /dev/null; then
  git fetch --no-tags origin "refs/heads/${branch}:refs/remotes/origin/${branch}"
  previous="$(git rev-parse "origin/${branch}")"
  git switch --detach "${previous}"
else
  git switch --detach "origin/${base}"
fi

for revision in "origin/${base}" refs/remotes/upstream/main; do
  if ! git merge --no-edit "${revision}"; then
    {
      echo 'Upstream synchronization requires conflict resolution. Main was not changed.'
      git diff --name-only --diff-filter=U
    } | tee -a "${GITHUB_STEP_SUMMARY:-/dev/stderr}"
    exit 1
  fi
done

changed=false
if [[ "$(git rev-parse HEAD)" != "${previous}" ]]; then
  git push origin "HEAD:refs/heads/${branch}"
  changed=true
fi

pull_request="$(gh pr list --repo "${repository}" --base "${base}" --head "${branch}" --state open --json number --jq '.[0].number // empty')"
body_file="$(mktemp)"
trap 'rm -f "${body_file}"' EXIT
if [[ -z ${pull_request} ]]; then
  cat > "${body_file}" << EOF
Merge new upstream commits while preserving the fork's commits.

Upstream: $(git rev-parse refs/remotes/upstream/main)
Validation is dispatched against this branch. Review and merge after checks pass.
Conflicts stop synchronization without changing main. This job never auto-merges.
EOF
  gh pr create --repo "${repository}" --base "${base}" --head "${branch}" \
    --title 'chore: merge upstream main' --body-file "${body_file}"
  changed=true
fi

if [[ ${changed} == true ]]; then
  gh workflow run "${check_workflow}" --repo "${repository}" --ref "${branch}"
fi
