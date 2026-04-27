#!/bin/bash --login
# Exit on any error
set -e

# Clean up
echo "Removing scratch directory ..."
rm -rf ~/scratch/uom-dissertation || true

echo "Copying updated code to scratch directory ..."
cp -r ~/GitRepos/uom-dissertation ~/scratch/

cd ~/scratch/uom-dissertation

echo "Final clean up to prepare scratch directory ..."
rm -rf .git/ doc/
rm README.md TODO.md
