#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# Load helpers
###############################################################################

: "${CREATE_RELEASE_COMMON_HELPER_SH:?CREATE_RELEASE_COMMON_HELPER_SH is not defined. Did you run: source set_env.sh ?}"
: "${TRACK_RELEASE_HELPER_SH:?TRACK_RELEASE_HELPER_SH is not defined. Did you run: source set_env.sh ?}"
: "${TRACK_MANIFEST_HELPER_SH:?TRACK_MANIFEST_HELPER_SH is not defined. Did you run: source set_env.sh ?}"

source "$CREATE_RELEASE_COMMON_HELPER_SH"
source "$TRACK_RELEASE_HELPER_SH"
source "$TRACK_MANIFEST_HELPER_SH"

setup_release_cleanup

###############################################################################
# Load configuration
###############################################################################

require_var TRACK_CONF
require_file "$TRACK_CONF"

source "$TRACK_CONF"

###############################################################################
# Build release
###############################################################################

validate_release_environment
map_track_formats
build_track_case_names

set_source_report_paths
set_source_vector_paths

preflight_release_sources

create_track_directories

copy_release_reports
copy_release_vectors

generate_track_params_pkg
generate_track_flists
generate_track_constraints
generate_release_manifest

###############################################################################
# Success
###############################################################################

mark_release_success

release_log "Track created successfully:"
release_log "  $TRACK_DIR"