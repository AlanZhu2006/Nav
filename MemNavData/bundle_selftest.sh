#!/usr/bin/env bash
# Validate a staged source bundle under node conditions BEFORE upload/submit.
#
# Catches locally, in seconds, the four failure classes that burned ~12 HPC
# submission cycles in 2026-08: missing transitive imports (depth_anything,
# kornia, requests), __pycache__ writes into read-only bundles, wrong Python
# version, and broken entry-script argument contracts.
#
# Usage:
#   bundle_selftest.sh <staging_root> <entries_file>
#
# entries_file: one check per line, three tab- or space-separated fields:
#   <python_interpreter>  import  <dotted.module>
#   <python_interpreter>  run     <relative/script.py --contract_dry_run ...>
# Blank lines and lines starting with # are ignored.
#
# The staging root is made read-only for the duration of the checks (mirrors
# the chmod a-w immutable-bundle convention) and restored afterwards.  Any
# check failure, or any __pycache__ created anywhere under the root, fails
# the whole selftest with a non-zero exit.
#
# Run it twice for full coverage: once locally, and once on the cluster login
# node (same script, remote interpreters) after rsync but before sbatch --
# the remote pass is what catches interpreter-environment gaps like the
# habitat env missing `requests`.
#
# SELFTEST_BUNDLE_SUBPATHS (optional): colon-separated staging-relative
# subdirectories appended to PYTHONPATH, mirroring job scripts that put both
# the bundle root and a package subdirectory on the path
# (e.g. SELFTEST_BUNDLE_SUBPATHS=MemNavData).
# SELFTEST_EXTRA_PYTHONPATH (optional): colon-separated, already-resolved
# external dependency paths.  This is used only when the production runner
# deliberately vendors a dependency outside the immutable source bundle, such
# as Habitat's ``pip/_vendor/requests`` fallback.  Callers must resolve and
# verify these paths before invoking the self-test.
set -uo pipefail

STAGING=${1:?usage: bundle_selftest.sh <staging_root> <entries_file>}
ENTRIES=${2:?usage: bundle_selftest.sh <staging_root> <entries_file>}
[[ -d "$STAGING" ]] || { echo "[selftest] no such staging root: $STAGING" >&2; exit 1; }
[[ -r "$ENTRIES" ]] || { echo "[selftest] no such entries file: $ENTRIES" >&2; exit 1; }
STAGING=$(cd "$STAGING" && pwd)
SELFTEST_PYTHONPATH="$STAGING"
_subpaths=${SELFTEST_BUNDLE_SUBPATHS:-}
for sub in ${_subpaths//:/ }; do
  SELFTEST_PYTHONPATH+=":$STAGING/$sub"
done
_external=${SELFTEST_EXTRA_PYTHONPATH:-}
if [[ -n "$_external" ]]; then
  SELFTEST_PYTHONPATH+=":$_external"
fi

failures=0
restore() { chmod -R u+w "$STAGING" 2>/dev/null || true; }
trap restore EXIT

find "$STAGING" -type d -name __pycache__ -prune -exec rm -rf -- {} + 2>/dev/null
chmod -R a-w "$STAGING"

while IFS= read -r line; do
  [[ -z "${line// }" || "${line#\#}" != "$line" ]] && continue
  read -r python kind target <<<"$line"
  if [[ ! -x "$python" ]]; then
    echo "[FAIL] interpreter missing: $python" >&2
    failures=$((failures + 1)); continue
  fi
  case "$kind" in
    import)
      if PYTHONPATH="$SELFTEST_PYTHONPATH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
          "$python" -c "import importlib; importlib.import_module('$target')" \
          >/dev/null 2>/tmp/selftest_err.$$; then
        echo "[PASS] import $target ($(basename "$(dirname "$(dirname "$python")")"))"
      else
        echo "[FAIL] import $target"; tail -3 /tmp/selftest_err.$$ >&2
        failures=$((failures + 1))
      fi
      ;;
    run)
      # shellcheck disable=SC2086
      if (cd "$STAGING" && PYTHONPATH="$SELFTEST_PYTHONPATH" PYTHONNOUSERSITE=1 \
          PYTHONDONTWRITEBYTECODE=1 "$python" $target) \
          >/dev/null 2>/tmp/selftest_err.$$; then
        echo "[PASS] run $target"
      else
        echo "[FAIL] run $target"; tail -3 /tmp/selftest_err.$$ >&2
        failures=$((failures + 1))
      fi
      ;;
    *)
      echo "[FAIL] unknown check kind '$kind' in: $line" >&2
      failures=$((failures + 1))
      ;;
  esac
done <"$ENTRIES"
rm -f /tmp/selftest_err.$$

stray=$(find "$STAGING" -type d -name __pycache__ | head -3)
if [[ -n "$stray" ]]; then
  echo "[FAIL] checks wrote __pycache__ into the bundle:" >&2
  echo "$stray" >&2
  failures=$((failures + 1))
fi

echo "[selftest] failures=$failures"
[[ "$failures" -eq 0 ]]
