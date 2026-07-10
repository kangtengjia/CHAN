#!/usr/bin/env bash
set -euo pipefail

TEXT_ENCODER="${TEXT_ENCODER:-bigru}"
exec bash "$(dirname "$0")/train_roma.sh" scanrefer "${TEXT_ENCODER}" "$@"
