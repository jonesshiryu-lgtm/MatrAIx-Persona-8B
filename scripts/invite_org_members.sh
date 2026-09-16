#!/usr/bin/env bash
# Batch-invite GitHub users to the MatrAIx-ai organization.
#
# Usage:
#   GH_TOKEN=<org-owner token with admin:org> ./scripts/invite_org_members.sh usernames.txt
#   ... where usernames.txt has one GitHub username per line (# comments allowed).
# Prints one line per user: "login: pending" (invited), "login: active" (already a member).
set -euo pipefail
file=${1:?usage: invite_org_members.sh usernames.txt}
while IFS= read -r u; do
  u=${u%%#*}; u=$(echo "$u" | tr -d '[:space:]')
  [ -z "$u" ] && continue
  gh api -X PUT "orgs/MatrAIx-ai/memberships/$u" -f role=member --jq '"\(.user.login): \(.state)"' \
    || echo "$u: FAILED" >&2
done < "$file"
