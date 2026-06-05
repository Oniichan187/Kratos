#!/usr/bin/env bash
# Kratos WSL Setup — run INSIDE WSL (wsl -e bash setup_wsl.sh)
# Installs Ollama with CUDA support for the RTX 4050 Laptop GPU.
set -euo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${CYAN}ℹ${NC}  $*"; }
success() { echo -e "${GREEN}✓${NC}  $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
error()   { echo -e "${RED}✗${NC}  $*"; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Kratos WSL + CUDA Setup        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════╝${NC}"
echo ""

# ── 1. Ollama ─────────────────────────────────────────────────────────────────
if command -v ollama &>/dev/null; then
    success "Ollama already installed: $(ollama --version 2>&1 | head -1)"
else
    info "Installing Ollama…"
    curl -fsSL https://ollama.com/install.sh | sh
    success "Ollama installed."
fi

# ── 2. NVIDIA / CUDA check ────────────────────────────────────────────────────
echo ""
if command -v nvidia-smi &>/dev/null; then
    success "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader | sed 's/^/    /'
    success "CUDA available — Ollama will use GPU acceleration."
else
    warn "nvidia-smi not found."
    warn "For CUDA support in WSL2 install the NVIDIA driver on Windows (≥ 525)."
    warn "The WSL2 CUDA toolkit is included automatically with the Windows driver."
    warn "After installing, restart WSL: wsl --shutdown"
fi

# ── 3. Verify Ollama can start ────────────────────────────────────────────────
echo ""
info "Test-starting Ollama (will stop after 5 s)…"
OLLAMA_HOST=127.0.0.1 ollama serve &>/tmp/ollama_test.log &
OLLAMA_PID=$!
sleep 5
if kill -0 "$OLLAMA_PID" 2>/dev/null; then
    success "Ollama starts successfully."
    kill "$OLLAMA_PID" 2>/dev/null || true
else
    warn "Ollama may have had a startup issue. Check /tmp/ollama_test.log"
fi

# ── 4. Done ───────────────────────────────────────────────────────────────────
echo ""
success "WSL setup complete."
echo ""
echo -e "  Next step (from Windows):"
echo -e "  ${CYAN}python setup_models.py${NC}"
echo ""
