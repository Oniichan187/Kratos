#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Kratos — WSL provisioning (runs INSIDE Ubuntu/WSL2).
#
# Normally you do NOT call this by hand — install.bat runs it for you. But it is
# fully idempotent, so running it again only fixes what is missing.
#
# It installs: build tools + Python, Ollama (GPU/CUDA via the Windows driver),
# starts the Ollama server, and pulls every abliterated model Kratos uses.
#
# CUDA note: in WSL2 you do NOT install a CUDA toolkit here. The NVIDIA *Windows*
# driver exposes the GPU to WSL automatically (/usr/lib/wsl/lib/libcuda.so).
# Ollama links against that. We only verify it is visible.
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${CYAN}i ${NC} $*"; }
ok()      { echo -e "${GREEN}OK${NC} $*"; }
warn()    { echo -e "${YELLOW}! ${NC} $*"; }
err()     { echo -e "${RED}x ${NC} $*"; }

# Models (all abliterated). Keep in sync with kratos/config.py.
PLANNER="huihui_ai/deepseek-r1-abliterated:8b-0528-qwen3-q4_K_M"   # planner + verifier
CODER="huihui_ai/qwen2.5-coder-abliterate:7b-instruct-q4_K_M"      # coder
EMBED="nomic-embed-text"                                            # vector knowledge base
# Compressor (kratos-planner) is built from the local GGUF by setup_models.py.

echo ""
echo -e "${CYAN}=== Kratos WSL provisioning (Ollama + CUDA + models) ===${NC}"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
info "Updating apt and installing base packages (python3, pip, curl, git)..."
sudo apt-get update -y -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    python3 python3-pip python3-venv curl ca-certificates git
ok "Base packages installed."

# ── 2. Ollama ─────────────────────────────────────────────────────────────────
echo ""
if command -v ollama &>/dev/null; then
    ok "Ollama already installed: $(ollama --version 2>&1 | head -1)"
else
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed."
fi

# ── 3. GPU / CUDA visibility ──────────────────────────────────────────────────
echo ""
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    ok "NVIDIA GPU visible inside WSL:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/     /'
    ok "Ollama will use CUDA acceleration."
elif [ -e /usr/lib/wsl/lib/libcuda.so.1 ]; then
    ok "WSL CUDA library present (/usr/lib/wsl/lib) — GPU acceleration available."
else
    warn "No GPU visible in WSL yet. Install the NVIDIA *Windows* driver (>= 535),"
    warn "then run 'wsl --shutdown' on Windows and re-run install.bat."
    warn "Kratos still works on CPU, just slower."
fi

# ── 4. Start the Ollama server ────────────────────────────────────────────────
echo ""
start_ollama() {
    if curl -fsS http://127.0.0.1:11434/api/version &>/dev/null; then return 0; fi
    if pgrep -x ollama &>/dev/null; then sleep 2; return 0; fi
    info "Starting Ollama server in the background..."
    nohup ollama serve >/tmp/kratos_ollama.log 2>&1 &
    for _ in $(seq 1 30); do
        sleep 1
        curl -fsS http://127.0.0.1:11434/api/version &>/dev/null && return 0
    done
    return 1
}
if start_ollama; then
    ok "Ollama server is up on http://127.0.0.1:11434"
else
    err "Ollama did not start. See /tmp/kratos_ollama.log"
    exit 1
fi

# ── 5. Pull the registry models ───────────────────────────────────────────────
pull() {
    local model="$1" label="$2"
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$model"; then
        ok "$label already present: $model"
    else
        info "Pulling $label: $model  (large download, be patient)..."
        ollama pull "$model"
        ok "$label ready."
    fi
}
echo ""
pull "$PLANNER" "Planner/Verifier"
pull "$CODER"   "Coder"
pull "$EMBED"   "Embeddings"

echo ""
ok "WSL provisioning complete."
echo -e "  Next (handled automatically by install.bat): ${CYAN}python setup_models.py${NC}"
echo -e "  (builds the 'kratos-planner' compressor from the bundled GGUF and saves config)"
echo ""
