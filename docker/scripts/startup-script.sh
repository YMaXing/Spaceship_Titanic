#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

/start-tracking-server.sh &
tail -F anything
