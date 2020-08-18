#!/bin/sh
export GPG_TTY=$(tty)
ls -la
gpg --quiet --batch --yes --decrypt --passphrase="$CONFIG_INI_PASSPHRASE" \
--output config.ini config.ini.gpg