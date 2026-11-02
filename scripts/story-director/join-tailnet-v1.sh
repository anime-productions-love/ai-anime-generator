#!/usr/bin/env bash
# Connect one ephemeral Anime Story Director runner through Tailscale OIDC.
# No Tailscale API auth-key creation and no stored TAILSCALE_AUTHKEY.
set -euo pipefail

: "${RUNNER_TEMP:?RUNNER_TEMP required}"
: "${TS_CLIENT_ID:?TS_OAUTH_CLIENT_ID is not configured}"
: "${TAILNET_TAG:?Tailscale federated device tag required}"
if [[ ! "$TAILNET_TAG" =~ ^tag:[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "::error::Expected a valid authorized Tailscale tag: tag:name" >&2
  exit 1
fi
: "${ACTIONS_ID_TOKEN_REQUEST_URL:?GitHub OIDC token URL unavailable; grant id-token: write}"
: "${ACTIONS_ID_TOKEN_REQUEST_TOKEN:?GitHub OIDC request token unavailable}"

version="1.104.2"
case "$(uname -m)" in
  x86_64) arch=amd64 ;;
  aarch64) arch=arm64 ;;
  *) echo "::error::Unsupported Tailscale runner architecture" >&2; exit 1 ;;
esac

cache_root="$RUNNER_TEMP/tailscale-static-v$version"
bin_dir="$cache_root/tailscale_${version}_$arch"
if [ ! -x "$bin_dir/tailscale" ] || [ ! -x "$bin_dir/tailscaled" ]; then
  mkdir -p "$cache_root"
  archive="$RUNNER_TEMP/tailscale_${version}_$arch.tgz"
  curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
    --retry 3 --max-time 150 \
    "https://pkgs.tailscale.com/stable/tailscale_${version}_$arch.tgz" \
    --output "$archive"
  tar -xzf "$archive" -C "$cache_root"
  rm -f "$archive"
  [ -x "$bin_dir/tailscale" ] && [ -x "$bin_dir/tailscaled" ] || {
    echo "::error::Tailscale distribution archive did not contain expected executables" >&2
    exit 1
  }
fi

sudo install -m 0755 "$bin_dir/tailscale" /usr/local/bin/tailscale
sudo install -m 0755 "$bin_dir/tailscaled" /usr/local/bin/tailscaled
sudo mkdir -p /var/run/tailscale

# In-memory state means no long-lived device or node credentials persist on disk.
# Keep the daemon running only for this ephemeral GitHub Actions job.
sudo /usr/local/bin/tailscaled --state=mem: \
  --socket=/var/run/tailscale/tailscaled.sock \
  > "$RUNNER_TEMP/browser-tailscaled-oidc.log" 2>&1 &
echo $! > "$RUNNER_TEMP/browser-tailscaled-oidc.pid"

ready=false
for attempt in $(seq 1 30); do
  if sudo /usr/local/bin/tailscale status --json 2>/dev/null | jq -e '.BackendState == "NeedsLogin" or .BackendState == "Running"' >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 1
done
if [ "$ready" != true ]; then
  echo "::error::Tailscale daemon did not become ready" >&2
  tail -n 20 "$RUNNER_TEMP/browser-tailscaled-oidc.log" | sed -E 's/(token|secret|key)=[^ ]+/\\1=[REDACTED]/g' >&2
  exit 1
fi

# Supported natively by Tailscale >= 1.94.0. The Tailscale daemon discovers
# the GitHub Actions OIDC JWT and requests a short-lived tagged node authorization.
# -E preserves GitHub's ACTIONS_ID_TOKEN_REQUEST_* environment for discovery.
echo "Connecting via native Tailscale OIDC as an authorized tagged ephemeral runner."
if ! timeout -k 5s 120s sudo -E /usr/local/bin/tailscale up \
  --client-id="$TS_CLIENT_ID" \
  --audience="api.tailscale.com/$TS_CLIENT_ID" \
  --advertise-tags="$TAILNET_TAG" \
  --accept-dns=false; then
  echo "::error::Native Tailscale OIDC registration failed; inspect the runner step and tailnet trust policy." >&2
  exit 1
fi

sudo /usr/local/bin/tailscale status --json | jq -e '.BackendState == "Running"' >/dev/null || {
  echo "::error::Tailscale connected without entering Running state" >&2
  exit 1
}

# Preserve prior action's ping-to-home exit-node readiness contract.
if [ -n "${TAILNET_PING:-}" ]; then
  timeout -k 5s 70s sudo /usr/local/bin/tailscale ping --until-direct=false --c=2 --timeout=10s "$TAILNET_PING" >/dev/null || {
    echo "::error::Tailscale established but home exit-node peer is unreachable" >&2
    exit 1
  }
fi
echo "Story Director joined isolated ephemeral tailnet via GitHub OIDC."
