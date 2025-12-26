#!/usr/bin/env bash

cat << EOM
WARNING: This script is not safe yet.
This setup script installs the CPU version of vLLM.
This script should be run after 'uv sync'.
This script should be run from the root of the project.
This script will overwrite any version of vllm installed in the project's virtual environment.
EOM

# for safety; exit if any command returns a non-zero exit code
set -e

# for good measure
uv sync --quiet

TEMP_DIR=$(mktemp -d)
echo "Cloning vLLM project to '$TEMP_DIR'..."
git clone --quiet --depth 1 https://github.com/vllm-project/vllm.git "$TEMP_DIR"
pushd "$TEMP_DIR" > /dev/null

echo "Installing vLLM CPU..."
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cpu \
VLLM_TARGET_DEVICE=cpu \
uv pip install . \
--python "$OLDPWD/.venv" \
--extra-index-url https://download.pytorch.org/whl/cpu \
--index-strategy unsafe-best-match \
--force-reinstall


echo "Cleaning up temporary directory '$TEMP_DIR'..."
popd > /dev/null
rm -rf "$TEMP_DIR"


