#!/bin/sh
export GPG_TTY=$(tty)
echo ls -la
echo $CONFIG_INI_PASSPHRASE
gpg --quiet --yes --decrypt --passphrase="$CONFIG_INI_PASSPHRASE" \
--output config.ini config.ini.gpg