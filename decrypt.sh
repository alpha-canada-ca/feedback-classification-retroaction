#!/bin/sh
export GPG_TTY=$(tty) 

echo "Listing current directory contents:"
ls -la ./config

# Decrypt the config.ini file into the config directory
gpg --quiet --batch --yes --decrypt --passphrase="$CONFIG_INI_PASSPHRASE" \
--output ./config/config.ini ./config/config.ini.gpg

# Decrypt the client_secret.json file into the config directory
gpg --quiet --batch --yes --decrypt --passphrase="$CONFIG_INI_PASSPHRASE" \
--output ./config/client_secret.json ./config/client_secret.json.gpg

echo "Listing config directory contents:"
ls -la ./config
