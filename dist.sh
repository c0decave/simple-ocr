#!/bin/sh
# Build a public release:
#   dist/simple-ocr-<version>.tar.gz         image, start.sh, compose.yaml, .env.example, README.txt
#   dist/simple-ocr-<version>-source.tar.gz  the committed source, without git history
# Only committed code goes in, built from "git archive HEAD": no .env, no saved runs, no images.
# Before anything is built the release is scanned for private data, and the build stops if any is found.
set -eu
cd "$(dirname "$0")"

if [ -n "$(git status --porcelain)" ]; then
    echo "Uncommitted changes: commit first, the package must match a commit." >&2
    exit 1
fi

version=$(cat VERSION)
commit=$(git rev-parse --short HEAD)
image="localhost/simple-ocr:${version}"
name="simple-ocr-${version}"
out="dist/${name}"
package="dist/${name}.tar.gz"

source=$(mktemp -d)
trap 'rm -rf "$source" "$source.hits" "$source.private-words" "$source.words" "$out"' EXIT
git archive HEAD | tar -x -C "$source"

# Private network addresses, mail addresses, private keys and home paths never go public, and neither
# do the words in .private-words (one per line, # for comments; git-ignored, so the list stays local).
private='(^|[^0-9])(10\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])|192\.168)\.[0-9]{1,3}\.[0-9]{1,3}([^0-9]|$)'
private="$private"'|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}'
private="$private"'|-----BEGIN [A-Z ]*PRIVATE KEY|/(home|mnt|Users)/'
# grep -l over the release: exit 0 = found, 1 = nothing found, anything else = the scan itself failed,
# and a failed scan stops the release instead of passing it.
scan() {
    status=0
    (cd "$source" && grep -rIl "$@" .) >> "$source.hits" || status=$?
    if [ "$status" -gt 1 ]; then
        echo "The private-data scan failed (grep exit $status), nothing was built." >&2
        exit 1
    fi
}
: > "$source.hits"
scan -E "$private"
if [ -f .private-words ]; then
    # Windows line ends and blanks around a word would make it never match
    sed -e 's/\r$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' .private-words > "$source.private-words"
    status=0   # exit 1 only means that every line is a comment
    grep -vE '^[[:space:]]*(#|$)' "$source.private-words" > "$source.words" || status=$?
    if [ "$status" -gt 1 ]; then
        echo "Could not read .private-words (grep exit $status), nothing was built." >&2
        exit 1
    fi
    if [ -s "$source.words" ]; then
        scan -iF -f "$source.words"
    fi
fi
if [ -s "$source.hits" ]; then
    echo "Private data in these files, nothing was built:" >&2
    sed -e 's#^\./##' "$source.hits" | sort -u | sed 's/^/  /' >&2
    exit 1
fi

sources="dist/${name}-source.tar.gz"
rm -rf "$out" "$package" "$package.sha256" "$sources" "$sources.sha256"
mkdir -p "$out"
podman build -q -t "$image" -f "$source/Containerfile" "$source" >/dev/null
podman save -q --format docker-archive -o "$out/image.tar" "$image"

for file in start.sh compose.yaml README.txt; do
    sed "s/__VERSION__/${version}/g" "$source/package/$file" > "$out/$file"
done
chmod 755 "$out/start.sh"
cp "$source/.env.example" "$source/VERSION" "$out/"
echo "$commit" > "$out/COMMIT"

tar -czf "$package.tmp" -C dist "$name"
mv "$package.tmp" "$package"
git archive --format=tar.gz --prefix="${name}-source/" -o "$sources" HEAD
(cd dist && sha256sum "${name}.tar.gz" > "${name}.tar.gz.sha256" && sha256sum "${name}-source.tar.gz" > "${name}-source.tar.gz.sha256")
echo "$package"
echo "$sources"
