#!/usr/bin/env bash
# Run a command with the S3 credentials from the gitignored .secrets.md (never printed, never stored).
# Usage: scripts/with-s3-creds.sh aws s3 ls s3://bucket/prefix/
set -euo pipefail
SECRETS="${MIRAW_SECRETS_FILE:-$(dirname "$0")/../.secrets.md}"
[ -f "$SECRETS" ] || { echo "missing $SECRETS (see docs/PLAN.md, section 4)" >&2; exit 1; }
AWS_ACCESS_KEY_ID=$(sed -n 's/^AWS_ACCESS_KEY_ID *= *//p' "$SECRETS" | tr -d '[:space:]')
AWS_SECRET_ACCESS_KEY=$(sed -n 's/^AWS_SECRET_KEY *= *//p' "$SECRETS" | tr -d '[:space:]')
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
unset AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_ARN AWS_REGION AWS_DEFAULT_REGION
export AWS_DEFAULT_REGION="${MIRAW_AWS_REGION:-eu-west-1}"
exec "$@"
