#!/usr/bin/env bash

git submodule update --init --single-branch  # full clone for tex submodule
# blobless clone binder submodules inside tex
(cd tex && git checkout -b tex && git pull origin tex)
# see https://github.blog/open-source/git/get-up-to-speed-with-partial-clone-and-shallow-clone/
git submodule update --init --filter=blob:none --recursive --single-branch --jobs $(nproc)

# setup branch tracking
git submodule foreach --recursive 'git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"; git checkout -b $(basename $PWD); git branch -u origin/$(basename $PWD); git fetch origin $(basename $PWD) --filter=blob:none; git config branch.$(basename $PWD).remote origin'
