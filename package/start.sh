#!/bin/sh
# Start simple-ocr from this package: load the image once, then run it with the settings from .env.
set -eu
cd "$(dirname "$0")"

image="localhost/simple-ocr:__VERSION__"
engine=${ENGINE:-}
if [ -z "$engine" ]; then
    if command -v podman >/dev/null 2>&1; then engine=podman
    elif command -v docker >/dev/null 2>&1; then engine=docker
    else
        echo "Neither podman nor docker found. Install one of them, or set ENGINE." >&2
        exit 1
    fi
fi

if [ ! -f .env ]; then
    echo "No .env found. Run: cp .env.example .env   and edit it." >&2
    exit 1
fi

if ! "$engine" image inspect "$image" >/dev/null 2>&1; then
    "$engine" load -i image.tar
fi

app_port=$(sed -n 's/^APP_PORT=//p' .env | tail -n 1 | tr -d ' \r')
app_port=${app_port:-8080}
# APP_BIND=0.0.0.0 makes the page reachable from other machines; the app has no login, so only on a trusted network.
app_bind=$(sed -n 's/^APP_BIND=//p' .env | tail -n 1 | tr -d ' \r')
app_bind=${app_bind:-127.0.0.1}
case "$app_bind" in
    \[*) ;;                          # already [IPv6]
    *:*) app_bind="[$app_bind]" ;;   # an IPv6 address needs brackets in -p
esac
echo "simple-ocr: http://${app_bind}:${app_port}"
# Finished benchmark runs are kept in the named volume simple-ocr-data (the same one compose.yaml uses).
exec "$engine" run --rm --name simple-ocr --env-file .env \
    -v simple-ocr-data:/data \
    -p "${app_bind}:${app_port}:8080" "$image"
