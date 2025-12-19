# .devcontainer/download_model.sh
#!/usr/bin/env bash
set -e

# Wenn Model schon da ist, nichts tun
if [ -f /model/Llama-3.1-8B-Instruct-bf16_q6_k.gguf ]; then
  echo "Model bereits in /model heruntergeladen, Download übersprungen."
  exit 0
fi

mkdir -p /model
echo "Downloade Sprachmodell Llama 3.1 8B Instruct…"
# Je nach Modell-Repo und Token
curl -L --create-dirs \
     https://huggingface.co/Mungert/Llama-3.1-8B-Instruct-GGUF/resolve/main/Llama-3.1-8B-Instruct-bf16_q6_k.gguf \
     -o /model/Llama-3.1-8B-Instruct-bf16_q6_k.gguf

echo "Download erfolgreich beendet."
