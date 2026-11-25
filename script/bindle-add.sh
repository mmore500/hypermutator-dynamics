#!/bin/bash

cd "$(dirname "$0")/.."

# Fetch all remote branches
git fetch --all --jobs $(nproc)
git fetch origin

mkdir -p bindle

# Loop through remote branches starting with 'bindle'
for branch in $(git branch -r | grep 'origin/bindle' | sed 's|origin/||'); do
    echo "Processing branch $branch"

    if [ -d bindle/$branch ]; then
        echo "Skipping branch $branch"
        continue
    fi

    # Add the branch as a submodule
    git clone --branch "$branch" --single-branch "https://github.com/mmore500/hypermutator-dynamics.git" "bindle/$branch" --jobs $(nproc) --depth 1
    git submodule add --branch "$branch" "https://github.com/mmore500/hypermutator-dynamics.git" "bindle/$branch"
    git config -f .gitmodules submodule."bindle/$branch".branch "$branch"
    git config -f .gitmodules submodule."bindle/$branch".shallow true

done
