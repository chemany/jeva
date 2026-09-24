#!/usr/bin/env bash
# Publish jeva to ModelScope.
#
#   1) put an access token where the SDK finds it (never in this repo):
#        mkdir -p ~/.modelscope
#        printf '{"access_token":"%s"}' '<YOUR_TOKEN>' > ~/.modelscope/credentials.json
#      token page: https://modelscope.cn/my/myaccesstoken
#
#   2) export MODELSCOPE_REPO=<namespace>/<name>   (default: imjasonli/jeva)
#
#   3) bash scripts/release_modelscope.sh
#
# (args verified against `modelscope create --help`)
# Uploads the merged transformer weights to the repo root, the GGUF variants under gguf/,
# and uses docs/hf-model-card.md as the repository README.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# ModelScope 的用户名与 GitHub 的 chemany 不是同一个账号
REPO=${MODELSCOPE_REPO:-imjasonli/jeva}
# v7 is the released revision; older runs stay on disk as MiniCPM5-2B-WebDecider-v{3..6}*
MERGED=${MERGED_DIR:-/root/code/models/MiniCPM5-2B-WebDecider-v7}
GGUF=${GGUF_DIR:-/root/code/models/MiniCPM5-2B-WebDecider-v7-GGUF}
PY=${PY:-python3}

# The SDK authenticates through MODELSCOPE_API_TOKEN, not through the file directly, so the
# token stays out of this repo and out of the shell history.
if [ -z "${MODELSCOPE_API_TOKEN:-}" ] && [ -f "$HOME/.modelscope/credentials.json" ]; then
  MODELSCOPE_API_TOKEN=$("$PY" -c 'import json,os;print(json.load(open(os.path.expanduser("~/.modelscope/credentials.json")))["access_token"])')
  export MODELSCOPE_API_TOKEN
fi
[ -n "${MODELSCOPE_API_TOKEN:-}" ] || {
  echo "No token. Export MODELSCOPE_API_TOKEN, or write ~/.modelscope/credentials.json" >&2
  echo '(see the header of this script).' >&2
  exit 1
}
[ -d "$MERGED" ] || { echo "merged model dir not found: $MERGED" >&2; exit 1; }
[ -d "$GGUF" ]   || { echo "gguf dir not found: $GGUF" >&2; exit 1; }

# The repo README is the ModelScope facing model card.
cp "$ROOT/docs/hf-model-card.md" "$MERGED/README.md"
cp "$ROOT/system_prompt.txt"     "$MERGED/system_prompt.txt" 2>/dev/null || true
cp "$ROOT/examples/quickstart.py" "$MERGED/example_client.py" 2>/dev/null || true

echo "▶ creating $REPO (ignored if it already exists)"
"$PY" -m modelscope.cli.cli create "$REPO" \
  --repo_type model \
  --visibility public \
  --license "Apache License 2.0" \
  --chinese_name "jeva 浏览器智能体决策模型" \
  --description "jeva — a 2B browser-agent decision model (MiniCPM5-2B fine-tune, merged weights)" \
  --base_model_id openbmb/MiniCPM5-2B \
  --model_source TRAINED_FROM_MODELSCOPE || true

echo "▶ uploading merged weights + model card"
"$PY" -m modelscope.cli.cli upload "$REPO" "$MERGED" \
  --commit-message "jeva: merged MiniCPM5-2B fine-tune (100% on the frozen suites, 100/100 tasks)"

echo "▶ uploading GGUF variants under gguf/"
"$PY" -m modelscope.cli.cli upload "$REPO" "$GGUF" gguf \
  --commit-message "jeva: F16 / Q8_0 / Q4_K_M GGUF"

cat <<EOF

done. 权重直链（供 jeva CLI 使用）：
  https://modelscope.cn/models/$REPO/resolve/master/gguf

自检：
  JEVA_CACHE=/tmp/jeva-check python3 -m jeva download --variant Q4_K_M
EOF
