#!/usr/bin/env bash
# setup.sh: create or check the project's Python environments for the behaviour-cloning backends.
#
#   .venv      numpy backend (bc-train --backend python), pins in requirements.txt. Every platform.
#   .venv-mlx  mlx backend (Apple GPU, --backend mlx), pins in requirements-mlx.txt. macOS 14+ on
#              Apple silicon only; skipped elsewhere.
#
# It only creates or updates these two venvs (or the dirs given with --venv-dir/--mlx-dir). It never
# deletes anything, never needs sudo and never touches a system Python. Run `./setup.sh -h` for options.
# The docs' hook is `./setup.sh --check || ./setup.sh`: check (read-only, no network), set up if needed.
#
# Numbers behind the checks (verified 2026-10-12 against PyPI metadata and pip on macOS arm64):
# - numpy 2.0.2 publishes CPython wheels for cp39-cp312 (requires_python >= 3.9); on macOS arm64 the
#   macosx_14_0 wheel links Accelerate (the reference) and the macosx_11_0 one OpenBLAS.
# - mlx 0.32.3 and mlx-metal 0.32.3: cp310-cp314 (and 3.13t/3.14t), macOS arm64 14_0/15_0/26_0 only;
#   numpy 2.5.4 requires_python >= 3.12. So the mlx venv needs CPython 3.12-3.14.
# - pip >= 20.3: pip 20.2.4 finds no numpy 2.0.2 wheel at all on macOS arm64 (no macOS 11+ arm64
#   tags), 20.3.4 picks the macosx_14_0_arm64 one. 20.3 also added PEP 600 manylinux_x_y tags.
#
# Written for bash 3.2 (macOS /bin/bash): no associative arrays, no empty "${arr[@]}" under set -u.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REQ_NUMPY=$ROOT/requirements.txt
REQ_MLX=$ROOT/requirements-mlx.txt
PIP_MIN=20.3
NP_LO=9;  NP_HI=12;  NP_PREF=3.9       # CPython 3.x minors with numpy 2.0.2 wheels; preferred minor
MLX_LO=12; MLX_HI=14; MLX_PREF=3.14.8  # CPython 3.x minors for mlx 0.32.3 + numpy 2.5.4; reference version
MLX_MACOS_MIN=14

usage() {
  cat <<'EOF'
Usage: ./setup.sh [options]

Sets up the project's Python venvs for the behaviour-cloning backends:
  .venv      numpy backend (bc-train --backend python), pins in requirements.txt
  .venv-mlx  mlx backend (Apple GPU, --backend mlx), pins in requirements-mlx.txt;
             macOS 14+ on Apple silicon only
A venv that already passes the check is left untouched. Nothing is ever deleted.

Options:
  --check             verify only: never installs, no network. Exit 0 when everything this
                      platform needs is usable, 1 with one line per problem otherwise.
                      Hook: ./setup.sh --check || ./setup.sh
  --numpy-only        only the numpy venv
  --mlx-only          only the mlx venv
  --recreate          move an existing venv aside to <dir>.bak-<UTC timestamp>, then create it
                      fresh (the backup is not deleted; it no longer runs once moved)
  --venv-dir DIR      numpy venv directory (default: <repo>/.venv)
  --mlx-dir DIR       mlx venv directory (default: <repo>/.venv-mlx)
  --python PATH       base interpreter for a new numpy venv (CPython 3.9-3.12; 3.9 preferred)
  --python-mlx PATH   base interpreter for a new mlx venv (CPython 3.12-3.14; 3.14.8 preferred)
  --verify            afterwards, run `node --test test/bc.test.js` against the venvs
  -h, --help          this help

Environment: without --venv-dir/--mlx-dir, a set PYTHON / PYTHON_MLX (the trainer's interpreter
overrides, used in git worktrees) is what gets checked; setup.sh does not install into those.
EOF
}

say()  { printf '%s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '  warning: %s\n' "$*"; }
note() { printf '  note: %s\n' "$*"; }
die()  { printf 'setup.sh: %s\n' "$*" >&2; exit 1; }
usage_error() { printf 'setup.sh: %s (see ./setup.sh --help)\n' "$*" >&2; exit 2; }

# Problems found by the current check_env call (count and text), and by the whole run.
CUR_N=0; CUR_TEXT=
ALL_N=0; ALL_TEXT=
problem() { printf '  problem: %s\n' "$*"; CUR_N=$((CUR_N + 1)); CUR_TEXT="${CUR_TEXT}$*"$'\n'; }
keep_problems() { # <label>: add the current check's problems to the run's
  local line
  [ "$CUR_N" -gt 0 ] || return 0
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    ALL_N=$((ALL_N + 1)); ALL_TEXT="${ALL_TEXT}$1: ${line}"$'\n'
  done <<<"$CUR_TEXT"
}

abspath() { case $1 in /*) p=$1 ;; *) p=$PWD/$1 ;; esac; [ "$p" = / ] || p=${p%/}; printf '%s\n' "$p"; }
ver_ge() { # <a> <b>: a >= b, comparing major.minor numerically
  local a1 a2 b1 b2
  a1=${1%%.*}; a2=${1#*.}; a2=${a2%%.*}; [ "$a2" != "$1" ] || a2=0
  b1=${2%%.*}; b2=${2#*.}; b2=${b2%%.*}; [ "$b2" != "$2" ] || b2=0
  case "$a1$a2$b1$b2" in *[!0-9]*) return 1 ;; esac
  [ "$a1" -gt "$b1" ] || { [ "$a1" -eq "$b1" ] && [ "$a2" -ge "$b2" ]; }
}

# Probes run with -I (isolated: no PYTHONPATH, no user site, no cwd on sys.path) and -B (no .pyc writes).
# One line: version, implementation, machine, free-threaded (0/1), sys.executable (last: may have spaces).
PY_INFO='import platform, sys, sysconfig
print("%d.%d.%d" % sys.version_info[:3], platform.python_implementation(), platform.machine(),
      int(bool(sysconfig.get_config_var("Py_GIL_DISABLED"))), sys.executable)'
# Prints one line per installed package that differs from a name==version pin in the file argv[1].
PY_PINS='import sys
from importlib.metadata import version, PackageNotFoundError
for raw in open(sys.argv[1]):
    line = raw.split("#", 1)[0].split(";", 1)[0].strip()
    if not line:
        continue
    name, sep, want = line.partition("==")
    name, want = name.strip(), want.strip()
    if not sep:
        print("%s: requirements line is not pinned with ==" % name)
        continue
    try:
        have = version(name)
    except PackageNotFoundError:
        print("%s is not installed (pin %s==%s)" % (name, name, want))
        continue
    if have != want:
        print("%s %s is installed, expected the pin %s==%s" % (name, have, name, want))'
PY_NUMPY='import numpy
try:
    blas = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
except Exception:
    blas = "unknown"
print(numpy.__version__, blas)'
PY_MLX='import mlx.core as mx
print(mx.__version__, mx.default_device())'

# ---- arguments -------------------------------------------------------------------------------
MODE=setup; NUMPY_ONLY=0; MLX_ONLY=0; RECREATE=0; VERIFY=0
VENV_DIR=; MLX_DIR=; BASE_PY=; BASE_PY_MLX=
while [ $# -gt 0 ]; do
  case $1 in
    --check) MODE=check ;;
    --numpy-only) NUMPY_ONLY=1 ;;
    --mlx-only) MLX_ONLY=1 ;;
    --recreate) RECREATE=1 ;;
    --verify) VERIFY=1 ;;
    --venv-dir|--mlx-dir|--python|--python-mlx)
      [ $# -ge 2 ] && [ -n "$2" ] || usage_error "$1 needs a value"
      case $1 in
        --venv-dir) VENV_DIR=$(abspath "$2") ;;
        --mlx-dir) MLX_DIR=$(abspath "$2") ;;
        --python) BASE_PY=$2 ;;
        --python-mlx) BASE_PY_MLX=$2 ;;
      esac
      shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage_error "unknown option: $1" ;;
  esac
  shift
done
[ "$NUMPY_ONLY$MLX_ONLY" != 11 ] || usage_error "--numpy-only and --mlx-only exclude each other"
[ "$MODE$RECREATE" != check1 ] || usage_error "--check never changes anything; drop --recreate"

# ---- platform --------------------------------------------------------------------------------
OS=$(uname -s); ARCH=$(uname -m); MACOS_VER=
if [ "$OS" = Darwin ]; then
  MACOS_VER=$(sw_vers -productVersion 2>/dev/null || echo 0)
  # A shell under Rosetta reports x86_64; the hardware (and the venvs) are arm64.
  if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = 1 ]; then ARCH=arm64; fi
fi
MLX_OK=0; MLX_WHY=
if [ "$OS" != Darwin ]; then MLX_WHY="mlx (Apple GPU) needs macOS on Apple silicon; this is $OS $ARCH"
elif [ "$ARCH" != arm64 ]; then MLX_WHY="mlx (Apple GPU) needs Apple silicon; this Mac is $ARCH"
elif ! ver_ge "$MACOS_VER" "$MLX_MACOS_MIN"; then MLX_WHY="mlx 0.32.3 needs macOS $MLX_MACOS_MIN+; this is macOS $MACOS_VER"
else MLX_OK=1; fi

DO_NUMPY=1; DO_MLX=$MLX_OK
[ "$MLX_ONLY" = 0 ] || DO_NUMPY=0
[ "$NUMPY_ONLY" = 0 ] || DO_MLX=0
if [ "$MLX_ONLY" = 1 ] && [ "$MLX_OK" = 0 ]; then die "--mlx-only: $MLX_WHY"; fi

# Targets: an explicit dir, else the trainer's env override (check only), else the repo default.
NP_DIR=; NP_PY=; NP_FROM_ENV=0
if [ -n "$VENV_DIR" ]; then NP_DIR=$VENV_DIR
elif [ -n "${PYTHON:-}" ]; then NP_PY=$PYTHON; NP_FROM_ENV=1
else NP_DIR=$ROOT/.venv; fi
[ -z "$NP_DIR" ] || NP_PY=$NP_DIR/bin/python
MLX_DIR_T=; MLX_PY=; MLX_FROM_ENV=0
if [ -n "$MLX_DIR" ]; then MLX_DIR_T=$MLX_DIR
elif [ -n "${PYTHON_MLX:-}" ]; then MLX_PY=$PYTHON_MLX; MLX_FROM_ENV=1
else MLX_DIR_T=$ROOT/.venv-mlx; fi
[ -z "$MLX_DIR_T" ] || MLX_PY=$MLX_DIR_T/bin/python

# ---- checks ----------------------------------------------------------------------------------
# Console scripts whose shebang names an interpreter that no longer exists (the venv was moved).
stale_shebangs() { # <venv dir>: prints the names, space-separated
  local f first interp out=
  for f in "$1"/bin/*; do
    [ -f "$f" ] && [ ! -L "$f" ] || continue
    first=
    IFS= read -r first <"$f" 2>/dev/null || [ -n "$first" ] || continue
    case $first in '#!'*) ;; *) continue ;; esac
    interp=${first#??}; interp=${interp# }; interp=${interp%% *}
    if [ "$interp" = /bin/sh ]; then # pip's wrapper for long paths: '''exec' "<python>" "$0" "$@"
      interp=$(sed -n "2s/^'''exec' \"\\([^\"]*\\)\".*/\\1/p" "$f")
      [ -n "$interp" ] || continue
    fi
    [ -x "$interp" ] || out="$out ${f##*/}"
  done
  printf '%s\n' "${out# }"
}

# check_env <numpy|mlx> <python> <venv dir or empty>
# Prints findings and sets CHK_STATE: ok | missing (no venv) | fixable (pip or packages; repaired in
# place by setup) | broken (interpreter unusable; needs --recreate or another interpreter).
check_env() {
  local kind=$1 py=$2 dir=$3 req lo hi out ver impl mach ft exe major rest minor pipv pipver stale
  CUR_N=0; CUR_TEXT=; CHK_STATE=ok
  if [ "$kind" = numpy ]; then req=$REQ_NUMPY; lo=$NP_LO; hi=$NP_HI; else req=$REQ_MLX; lo=$MLX_LO; hi=$MLX_HI; fi
  if [ -n "$dir" ] && [ ! -e "$dir" ]; then
    problem "no $kind venv at $dir (fix: ./setup.sh)"; CHK_STATE=missing; return 0
  fi
  if [ -n "$dir" ] && [ ! -e "$py" ] && [ ! -L "$py" ]; then
    problem "$dir exists but has no bin/python, so it is not a venv (fix: ./setup.sh --recreate moves it aside)"
    CHK_STATE=broken; return 0
  fi
  if ! out=$("$py" -I -B -c "$PY_INFO" 2>&1); then
    problem "$py does not run (its base interpreter was moved or removed?): $(printf '%s\n' "$out" | head -n 1) (fix: ./setup.sh --recreate)"
    CHK_STATE=broken; return 0
  fi
  read -r ver impl mach ft exe <<<"$out"
  major=${ver%%.*}; rest=${ver#*.}; minor=${rest%%.*}
  info "interpreter: $impl $ver ($mach) $py"
  if [ "$impl" != CPython ] || [ "$major" != 3 ] || [ "$minor" -lt "$lo" ] || [ "$minor" -gt "$hi" ]; then
    problem "$py is $impl $ver; the $kind venv needs CPython 3.$lo-3.$hi, the versions its pinned wheels exist for (fix: ./setup.sh --recreate)"
    CHK_STATE=broken; return 0
  fi
  if [ "$mach" != "$ARCH" ]; then
    if [ "$kind" = mlx ]; then
      problem "$py is a $mach interpreter on $ARCH hardware; mlx needs a native arm64 Python (fix: ./setup.sh --recreate --python-mlx <arm64 python>)"
      CHK_STATE=broken; return 0
    fi
    warn "$py is a $mach interpreter on $ARCH hardware (Rosetta?); it works, but it is not the reference engine"
  fi
  if [ "$kind" = numpy ] && [ "$major.$minor" != "$NP_PREF" ]; then
    warn "Python $ver works, but $NP_PREF is preferred: it is the reference .venv and exercises the rule that trainer/bc_common.py stays 3.9-compatible"
  fi
  if [ "$kind" = mlx ] && [ "$ver" != "$MLX_PREF" ]; then
    warn "Python $ver works, but the reference mlx venv is $MLX_PREF (pyenv)"
  fi
  [ "$ft" = 0 ] || warn "$py is a free-threaded build; it works, but it is not the reference engine"

  if ! pipv=$("$py" -I -B -m pip --version --disable-pip-version-check 2>/dev/null); then
    problem "pip is not available to $py (fix: ./setup.sh)"; CHK_STATE=fixable
  else
    pipver=$(printf '%s\n' "$pipv" | awk '{print $2}')
    if ver_ge "$pipver" "$PIP_MIN"; then info "pip $pipver"
    else problem "pip $pipver is older than $PIP_MIN, the first pip that knows the macOS 11+ arm64 and PEP 600 manylinux wheel tags (fix: ./setup.sh)"; CHK_STATE=fixable; fi
  fi

  if ! out=$("$py" -I -B -c "$PY_PINS" "$req" 2>&1); then
    problem "could not read the installed versions: $(printf '%s\n' "$out" | tail -n 1) (fix: ./setup.sh)"; CHK_STATE=fixable
  elif [ -n "$out" ]; then
    while IFS= read -r line; do problem "$line (fix: ./setup.sh)"; done <<<"$out"
    CHK_STATE=fixable
  else
    info "pins match ${req##*/}"
    if [ "$kind" = numpy ]; then
      if out=$("$py" -I -B -c "$PY_NUMPY" 2>&1); then
        info "numpy ${out%% *}, BLAS ${out#* }"
        if [ "$OS/$ARCH" = Darwin/arm64 ] && [ "${out#* }" != accelerate ]; then
          warn "numpy links ${out#* }, not Apple Accelerate (needs macOS 14+); test/bc.test.js's python param pins assume Accelerate"
        fi
      else
        problem "import numpy fails: $(printf '%s\n' "$out" | tail -n 1) (fix: ./setup.sh --recreate)"; CHK_STATE=fixable
      fi
    else
      if out=$("$py" -I -B -c "$PY_MLX" 2>&1); then
        info "mlx ${out%% *}, default device ${out#* }"
        case $out in *gpu*) ;; *) warn "mlx's default device is not the GPU" ;; esac
      else
        problem "import mlx.core fails: $(printf '%s\n' "$out" | tail -n 1) (fix: ./setup.sh --recreate)"; CHK_STATE=fixable
      fi
    fi
  fi

  if [ -z "$dir" ] && [ -f "$(dirname -- "$(dirname -- "$py")")/pyvenv.cfg" ]; then dir=$(dirname -- "$(dirname -- "$py")"); fi
  if [ -n "$dir" ]; then
    stale=$(stale_shebangs "$dir")
    if [ -n "$stale" ]; then
      warn "console scripts in $dir/bin point at an interpreter that no longer exists (the venv was moved): $stale."
      info "         Harmless: the trainer runs bin/python directly. Run pip as $dir/bin/python -m pip, or ./setup.sh --recreate."
    fi
  fi
  return 0
}

# ---- setup -----------------------------------------------------------------------------------
# try_base <kind> <candidate>: sets BASE/BASE_VER when the candidate runs and fits the kind's range.
REJECTED=
try_base() {
  local kind=$1 cand=$2 out ver impl mach ft exe rest minor lo hi
  if [ "$kind" = numpy ]; then lo=$NP_LO; hi=$NP_HI; else lo=$MLX_LO; hi=$MLX_HI; fi
  command -v "$cand" >/dev/null 2>&1 || return 1
  out=$("$cand" -I -B -c "$PY_INFO" 2>/dev/null) || { REJECTED="$REJECTED; $cand (does not run)"; return 1; }
  read -r ver impl mach ft exe <<<"$out"
  rest=${ver#*.}; minor=${rest%%.*}
  if [ "$impl" != CPython ] || [ "${ver%%.*}" != 3 ] || [ "$minor" -lt "$lo" ] || [ "$minor" -gt "$hi" ]; then
    REJECTED="$REJECTED; $cand ($impl $ver)"; return 1
  fi
  if [ "$kind" = mlx ] && [ "$mach" != arm64 ]; then REJECTED="$REJECTED; $cand ($mach, not arm64)"; return 1; fi
  BASE=$exe; BASE_VER=$ver
}

# find_base <kind>: sets BASE/BASE_VER or exits with the reason.
find_base() {
  local kind=$1 c pfx
  BASE=; BASE_VER=; REJECTED=
  if [ "$kind" = numpy ]; then
    if [ -n "$BASE_PY" ]; then
      try_base numpy "$BASE_PY" || die "refusing --python $BASE_PY:${REJECTED#;}${REJECTED:+.} The numpy venv needs CPython 3.$NP_LO-3.$NP_HI (numpy 2.0.2 publishes wheels for cp39-cp312 only)"
      return 0
    fi
    for c in python3.9 python3.10 python3.11 python3.12; do try_base numpy "$c" && return 0; done
    # /usr/bin/python3 on a Mac without the Command Line Tools opens an install dialog: skip it there.
    if [ "$OS" != Darwin ] || xcode-select -p >/dev/null 2>&1; then try_base numpy /usr/bin/python3 && return 0; fi
    try_base numpy python3 && return 0
    die "no CPython 3.$NP_LO-3.$NP_HI found for the numpy venv (rejected:${REJECTED#;}). Install one (macOS: xcode-select --install gives 3.9; Debian/Ubuntu: apt install python3 python3-venv), or pass --python PATH"
  fi
  if [ -n "$BASE_PY_MLX" ]; then
    try_base mlx "$BASE_PY_MLX" || die "refusing --python-mlx $BASE_PY_MLX:${REJECTED#;}${REJECTED:+.} The mlx venv needs an arm64 CPython 3.$MLX_LO-3.$MLX_HI (mlx 0.32.3 wheels: cp310-cp314; numpy 2.5.4: >= 3.12)"
    return 0
  fi
  if command -v pyenv >/dev/null 2>&1; then
    if pfx=$(pyenv prefix "$MLX_PREF" 2>/dev/null) && [ -n "$pfx" ]; then
      try_base mlx "$pfx/bin/python${MLX_PREF%.*}" && return 0
    else
      note "pyenv has no Python $MLX_PREF (the reference for the mlx venv). To use it, run: pyenv install $MLX_PREF"
    fi
  fi
  for c in python3.14 python3.13 python3.12 python3; do try_base mlx "$c" && return 0; done
  die "no arm64 CPython 3.$MLX_LO-3.$MLX_HI found for the mlx venv (rejected:${REJECTED#;}). Install one, e.g. pyenv install $MLX_PREF, or pass --python-mlx PATH"
}

# install_into <kind> <python>: bring pip to the minimum, then install the pins (wheels only).
install_into() {
  local kind=$1 py=$2 req pipver
  if [ "$kind" = numpy ]; then req=$REQ_NUMPY; else req=$REQ_MLX; fi
  if ! "$py" -I -m pip --version --disable-pip-version-check >/dev/null 2>&1; then
    info "pip missing: python -m ensurepip"
    "$py" -I -m ensurepip --upgrade --default-pip >/dev/null || { problem "ensurepip failed in $py"; return 1; }
  fi
  pipver=$("$py" -I -m pip --version --disable-pip-version-check | awk '{print $2}')
  if ! ver_ge "$pipver" "$PIP_MIN"; then
    info "pip $pipver < $PIP_MIN: upgrading pip inside the venv"
    "$py" -I -m pip install --disable-pip-version-check --only-binary=:all: --upgrade "pip>=$PIP_MIN" \
      || { problem "could not upgrade pip in $py (no network?)"; return 1; }
  fi
  info "pip install -r ${req##*/} (binary wheels only)"
  if ! "$py" -I -m pip install --disable-pip-version-check --only-binary=:all: -r "$req"; then
    problem "pip could not install the pins in ${req##*/} for $(cd / && "$py" -I -c 'import platform; print("Python", platform.python_version(), "on", platform.system(), platform.machine())'): no network/PyPI, or no binary wheel for this Python and platform (source builds are disabled on purpose)"
    return 1
  fi
}

FAILED=0
# setup_env <kind> <python> <dir> <from env 0/1>
setup_env() {
  local kind=$1 py=$2 dir=$3 from_env=$4 ts bak
  if [ "$from_env" = 1 ]; then
    if [ "$RECREATE" = 1 ]; then
      problem "--recreate: the $kind interpreter comes from \$$( [ "$kind" = numpy ] && echo PYTHON || echo PYTHON_MLX )=$py, which setup.sh does not own; pass --$( [ "$kind" = numpy ] && echo venv || echo mlx )-dir DIR or unset it"
      keep_problems "$kind"; FAILED=1; return 0
    fi
    check_env "$kind" "$py" ""
    if [ "$CHK_STATE" = ok ]; then info "=> usable, left untouched"; return 0; fi
    keep_problems "$kind"; FAILED=1
    info "=> setup.sh only installs into the project's venv dirs: run it in the checkout that owns $py, or unset $( [ "$kind" = numpy ] && echo PYTHON || echo PYTHON_MLX )"
    return 0
  fi
  if [ "$RECREATE" = 1 ] && { [ -e "$dir" ] || [ -L "$dir" ]; }; then
    ts=$(date -u +%Y%m%dT%H%M%SZ); bak=$dir.bak-$ts
    [ ! -e "$bak" ] || { problem "$bak already exists; not overwriting it"; keep_problems "$kind"; FAILED=1; return 0; }
    mv -- "$dir" "$bak"
    info "--recreate: moved $dir to $bak (not deleted; it no longer runs from there)"
  fi
  check_env "$kind" "$py" "$dir"
  case $CHK_STATE in
    ok) info "=> up to date, left untouched"; return 0 ;;
    broken) keep_problems "$kind"; FAILED=1; info "=> not repairing in place; run ./setup.sh --recreate (moves it aside, never deletes)"; return 0 ;;
    missing)
      find_base "$kind"
      info "creating $dir from $BASE (Python $BASE_VER)"
      if ! "$BASE" -I -c 'import venv, ensurepip' >/dev/null 2>&1; then
        if [ "$OS" = Linux ]; then
          die "$BASE cannot create venvs (no venv/ensurepip module). On Debian/Ubuntu/WSL: sudo apt install python${BASE_VER%.*}-venv, then rerun ./setup.sh"
        fi
        die "$BASE cannot create venvs (no venv/ensurepip module); pass another interpreter with $( [ "$kind" = numpy ] && echo --python || echo --python-mlx )"
      fi
      "$BASE" -m venv "$dir" || die "$BASE -m venv $dir failed" ;;
    fixable) info "=> repairing in place" ;;
  esac
  if ! install_into "$kind" "$py"; then keep_problems "$kind"; FAILED=1; return 0; fi
  say "  re-checking:"
  check_env "$kind" "$py" "$dir"
  if [ "$CHK_STATE" != ok ]; then keep_problems "$kind"; FAILED=1; fi
}

# ---- main ------------------------------------------------------------------------------------
case "$OS/$ARCH" in
  Darwin/arm64) PLAT="macOS $MACOS_VER arm64" ;;
  Darwin/*) PLAT="macOS $MACOS_VER $ARCH" ;;
  *) PLAT="$OS $ARCH" ;;
esac
WHAT=; [ "$DO_NUMPY" = 0 ] || WHAT="numpy venv"; [ "$DO_MLX" = 0 ] || WHAT="${WHAT:+$WHAT + }mlx venv"
say "setup.sh ($MODE) on $PLAT: $WHAT"
if [ "$MLX_OK" = 0 ] && [ "$MLX_ONLY" = 0 ]; then note "skipping the mlx venv: $MLX_WHY. The numpy backend covers everything."; fi
if [ "$DO_NUMPY" = 1 ] && [ "$OS/$ARCH" != Darwin/arm64 ]; then
  note "numpy backend results are recorded on macOS arm64 (numpy 2.0.2 + Accelerate); with OpenBLAS here they are not expected to be bit-identical, and test/bc.test.js skips its python param pins on this platform"
fi

if [ "$DO_NUMPY" = 1 ]; then
  say "numpy venv (${NP_DIR:-PYTHON=$NP_PY}):"
  if [ "$MODE" = check ]; then check_env numpy "$NP_PY" "$NP_DIR"; keep_problems numpy
  else setup_env numpy "$NP_PY" "$NP_DIR" "$NP_FROM_ENV"; fi
fi
if [ "$DO_MLX" = 1 ]; then
  say "mlx venv (${MLX_DIR_T:-PYTHON_MLX=$MLX_PY}):"
  if [ "$MODE" = check ]; then check_env mlx "$MLX_PY" "$MLX_DIR_T"; keep_problems mlx
  else setup_env mlx "$MLX_PY" "$MLX_DIR_T" "$MLX_FROM_ENV"; fi
fi

if [ "$ALL_N" -gt 0 ]; then
  say ""
  if [ "$MODE" = check ]; then say "setup.sh --check: $ALL_N problem(s):"; else say "setup.sh: $ALL_N problem(s) left:"; fi
  printf '%s' "$ALL_TEXT" | sed 's/^/  - /'
  exit 1
fi
[ "$FAILED" = 0 ] || exit 1

say ""
say "OK. In a git worktree (which has no venvs of its own), point the trainer at these:"
[ "$DO_NUMPY" = 0 ] || say "  export PYTHON=$NP_PY"
[ "$DO_MLX" = 0 ] || say "  export PYTHON_MLX=$MLX_PY"

if [ "$VERIFY" = 1 ]; then
  if ! command -v node >/dev/null 2>&1; then note "--verify: node not found; skipping test/bc.test.js"; exit 0; fi
  say ""
  say "--verify: node --test test/bc.test.js"
  set --
  [ "$DO_NUMPY" = 0 ] || set -- "PYTHON=$NP_PY"
  [ "$DO_MLX" = 0 ] || set -- ${1+"$@"} "PYTHON_MLX=$MLX_PY"
  ( cd -- "$ROOT" && env ${1+"$@"} node --test test/bc.test.js ) || die "--verify: test/bc.test.js failed"
  say "--verify: test/bc.test.js passed"
fi
