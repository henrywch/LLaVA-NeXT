#!/bin/bash

# --- config ---
USERNAME=henrywch
HOST=bgithub.xyz
CREDENTIALS_FILE="$HOME/.git-credentials"

# --- procedure ---
echo -n "${USERNAME}@${HOST} Fine-Grained Token (input will be invisible): "
read -s NEW_TOKEN
echo

if [ -z "$NEW_TOKEN" ]; then
    echo -e "Error: Fine-Grained Token Empty, Cancelled." >&2
    exit 1
fi

NEW_LINE="https://${USERNAME}:${NEW_TOKEN}@${HOST}"
SEARCH_PATTERN="^https://${USERNAME}:.*@${HOST}$"

touch "$CREDENTIALS_FILE"

if grep -qE "$SEARCH_PATTERN" "$CREDENTIALS_FILE"; then
    sed "s|${SEARCH_PATTERN}|${NEW_LINE}|" "$CREDENTIALS_FILE" > "${CREDENTIALS_FILE}.tmp" && mv "${CREDENTIALS_FILE}.tmp" "$CREDENTIALS_FILE"
    echo -e "Success: Fine-grained Token Updated in ${CREDENTIALS_FILE}."
else
    echo "$NEW_LINE" >> "$CREDENTIALS_FILE"
    echo -e "Success: Fine-grained Token Added to ${CREDENTIALS_FILE}."
fi

if [ "$(uname)" != "Darwin" ]; then
    CURRENT_PERMS=$(stat -c "%a" "$CREDENTIALS_FILE")
    if [ "$CURRENT_PERMS" != "600" ]; then
        echo "Warning: It is recommended to set the file permissions of ${CREDENTIALS_FILE} to 600 to protect your credentials."
        echo "You can run the command: chmod 600 ${CREDENTIALS_FILE}"
    fi
fi

exit 0