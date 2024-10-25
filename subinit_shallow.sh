#!/usr/bin/env bash

git submodule update --init --single-branch  # full clone for tex submodule
# blobless clone binder submodules inside tex
(cd tex && git pull origin tex)
# see https://github.blog/open-source/git/get-up-to-speed-with-partial-clone-and-shallow-clone/
git submodule update --init --filter=blob:none --recursive --single-branch --jobs $(nproc)