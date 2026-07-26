#!/usr/bin/env bash
# Cut a release: bump the version, tag it, push it, and publish the notes.
#
#   ./scripts/release.sh 0.4.0            # tag and push
#   ./scripts/release.sh 0.4.0 --dry-run  # show what it would do
#
# Expects CHANGELOG.md to already have a section for the version — writing the
# notes is the part a human should do, so this refuses to invent them.
set -euo pipefail

VERSION="${1:?usage: release.sh X.Y.Z [--dry-run]}"
DRY_RUN="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INIT="$ROOT/llamactl/__init__.py"
CHANGELOG="$ROOT/CHANGELOG.md"

[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "version must look like 1.2.3" >&2; exit 1; }

cd "$ROOT"
[[ -z "$(git status --porcelain)" ]] || { echo "working tree is dirty — commit first" >&2; exit 1; }
git rev-parse "v$VERSION" >/dev/null 2>&1 && { echo "tag v$VERSION already exists" >&2; exit 1; }
grep -q "^## \[$VERSION\]" "$CHANGELOG" || {
  echo "CHANGELOG.md has no '## [$VERSION]' section — write the notes first" >&2; exit 1; }

# The notes for this version: everything between its heading and the next one.
# Matched with index() rather than a regex — awk would read the brackets in
# "## [0.3.0]" as a character class and quietly match nothing.
NOTES="$(awk -v tag="## [$VERSION]" '
  index($0, tag) == 1 {found = 1; next}
  found && index($0, "## [") == 1 {exit}
  found {print}' "$CHANGELOG")"

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "would set __version__ = \"$VERSION\", tag v$VERSION and push"
  echo "--- notes ---"; echo "$NOTES"
  exit 0
fi

sed -i.bak "s/^__version__ = .*/__version__ = \"$VERSION\"/" "$INIT" && rm -f "$INIT.bak"
git add "$INIT" "$CHANGELOG"
git commit -m "Release v$VERSION"
git tag -a "v$VERSION" -m "v$VERSION"
git push origin HEAD --follow-tags

if command -v gh >/dev/null; then
  printf '%s\n' "$NOTES" | gh release create "v$VERSION" --title "v$VERSION" --notes-file -
  echo "published https://github.com/p-jungjitdamrong/llama.cpp-controller/releases/tag/v$VERSION"
else
  echo "tag pushed; install gh to publish release notes automatically"
fi
