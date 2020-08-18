#!/bin/sh
export GPG_TTY=$(tty)
gpg --quiet --yes --decrypt --passphrase="$CONFIG_INI_PASSPHRASE" \
--output config.ini config.ini.gpg