#!/bin/bash
echo "Looking for .out files to delete..."

# First, list the files that will be deleted
echo "The following files will be permanently deleted:"
ls -l *.out

# Ask for user confirmation before deleting
read -p "Are you sure you want to delete these files? (y/n) " -n 1 -r
echo # move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "Deleting files..."
  rm *.out
  echo "Done."
else
  echo "Deletion cancelled."
fi
