#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[create_release_common.sh] ERROR: this script must be sourced, not executed." >&2
    exit 1
fi

release_log() {
    echo "[create_release.sh] $*"
}

die() {
    echo "[create_release.sh] ERROR: $*" >&2
    exit 1
}

require_var() {
    local name="$1"

    if [[ -z "${!name:-}" ]]; then
        die "required variable '$name' is undefined or empty"
    fi
}

require_file() {
    local path="$1"

    if [[ ! -f "$path" ]]; then
        die "missing file: $path"
    fi
}

copy_file_as() {
    local src="$1"
    local dst="$2"

    require_file "$src"
    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst"
}

copy_file_to_dir() {
    local src="$1"
    local dst_dir="$2"

    require_file "$src"
    mkdir -p "$dst_dir"
    cp -f "$src" "$dst_dir/"
}

get_next_revision() {
    local track_base="$1"
    local last_rev=0

    for d in "${track_base}".rev*; do
        [[ -e "$d" ]] || continue

        local rev_part="${d##*.rev}"

        if [[ "$rev_part" =~ ^[0-9]+$ ]]; then
            if (( rev_part > last_rev )); then
                last_rev="$rev_part"
            fi
        fi
    done

    echo $((last_rev + 1))
}

TRACK_CREATED=0

cleanup_failed_track() {
    local status=$?

    if [[ "$TRACK_CREATED" -eq 1 && -n "${TRACK_DIR:-}" && -d "$TRACK_DIR" ]]; then
        echo "[create_release.sh] ERROR: release failed. Removing incomplete track: $TRACK_DIR" >&2
        rm -rf "$TRACK_DIR"
    fi

    exit "$status"
}

setup_release_cleanup() {
    TRACK_CREATED=0
    trap cleanup_failed_track ERR INT
}

mark_track_created() {
    TRACK_CREATED=1
}

mark_release_success() {
    TRACK_CREATED=0
    trap - ERR INT
}