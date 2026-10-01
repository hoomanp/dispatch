#!/bin/bash
# Run on your always-on Ollama node. Exposes Ollama on the network and
# pulls the models DISPATCH uses for chat, memory, and embeddings.
set -e
echo "Expose Ollama on the network:"
echo "  macOS: launchctl setenv OLLAMA_HOST 0.0.0.0:11434 && brew services restart ollama"
echo "  Linux: OLLAMA_HOST=0.0.0.0:11434 ollama serve"
echo
ollama pull phi4-mini
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
echo "Done."
