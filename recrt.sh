#!/bin/bash

# A script to safely delete the first line of the ~/.git-credentials file.

# Define the path to the credentials file
CRED_FILE="$HOME/.git-credentials"

# Check if the file actually exists
if [ ! -f "$CRED_FILE" ]; then
	    echo "Error: The file $CRED_FILE does not exist."
	        exit 1
fi

# Use sed to delete the first line ('1d').
# The -i.bak flag edits the file in-place and creates a backup named ~/.git-credentials.bak
# This is a safety measure in case something goes wrong.
sed -i.bak '1d' "$CRED_FILE"

echo "Successfully deleted the first line of $CRED_FILE."
echo "A backup of the original file has been saved as $CRED_FILE.bak"

# Optional: Show the content of the file after deletion
echo "Current content of the file:"
cat "$CRED_FILE"

