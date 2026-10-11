#!/usr/bin/env bash
# Stamp the build with its git commit (Resources/build.json, saved in every recording), then build, sign and
# install the release app with xtool. Run in WSL / Linux from anywhere; extra arguments go to xtool:
#
#   scripts/build.sh                   # your phone
#   scripts/build.sh --udid <UDID>     # another phone (see `xtool devices`)
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
g() { git -c safe.directory='*' -C "$here" "$@"; }  # the repo is on the Windows drive, owned by another user id
commit="$(g rev-parse HEAD 2>/dev/null || echo unknown)"
dirty=false
if [ -n "$(g status --porcelain 2>/dev/null)" ]; then dirty=true; fi
mkdir -p "$here/Resources"
printf '{"commit": "%s", "dirty": %s, "built": "%s"}\n' "$commit" "$dirty" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$here/Resources/build.json"
echo "build ${commit:0:7}$([ "$dirty" = true ] && echo ' (with uncommitted changes)')"
cd "$here"
exec xtool dev run -c release "$@"
