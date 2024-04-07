#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

cp -r C:/Users/garru/.ssh .
ssh-add .ssh/id_ed25519

/start-tracking-server.sh &
tail -F anything
