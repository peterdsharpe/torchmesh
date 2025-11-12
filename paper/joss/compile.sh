#!/usr/bin/env bash
# Compile JOSS paper using the official Open Journals Docker image

sudo docker run --rm \
    --volume "$PWD:/data" \
    --user "$(id -u):$(id -g)" \
    --env JOURNAL=joss \
    openjournals/inara