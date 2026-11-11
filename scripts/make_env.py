#!/usr/bin/env python3
"""Create ``.env`` from ``.env.example`` with two generated passwords. `make setup` runs this.

MariaDB's Docker image sets the root password and the application user's password when it
first initialises its data volume, from the two variables in ``.env``. This script writes that
file once, so a first `make setup` needs no editing:

- an existing ``.env`` is never touched (exit 0, one line saying so);
- otherwise ``.env.example`` is copied line by line, and the values of
  ``WIKILENSE_DB_PASSWORD`` and ``WIKILENSE_DB_ROOT_PASSWORD`` are replaced by two different
  random strings of 32 hexadecimal characters (128 bits each, from ``secrets``);
- the new file is readable by its owner only (mode 0600 where the system has file modes);
- the passwords are never printed.

To choose the passwords yourself, copy ``.env.example`` to ``.env`` and edit it before the first
`make setup`. Standard library only, and no syntax newer than Python 3.6, so the script also
runs under the interpreter that `make setup` is about to reject as too old.

usage: python3 scripts/make_env.py [--example .env.example] [--out .env]
"""

import argparse
import os
import re
import secrets
import sys

PASSWORD_VARIABLES = ("WIKILENSE_DB_PASSWORD", "WIKILENSE_DB_ROOT_PASSWORD")
#: ``NAME=value`` for one of the two variables, with the optional ``export`` that dotenv allows.
ASSIGNMENT = re.compile(
    r"^(?P<head>\s*(?:export\s+)?(?P<name>" + "|".join(PASSWORD_VARIABLES) + r")\s*=).*$"
)
PASSWORD_BYTES = 16  # 32 hexadecimal characters: no quoting needed by Docker Compose or dotenv


class EnvError(Exception):
    """The example file is missing or does not define both password variables."""


def render(example_text, new_password=None):
    """Return the text of ``.env``: ``example_text`` with both password values replaced.

    ``new_password`` is a zero-argument function returning one password; the default draws
    ``PASSWORD_BYTES`` random bytes as hexadecimal. Every other line is kept as it is.
    Raises EnvError when a password variable is not assigned in ``example_text``.
    """
    if new_password is None:
        def new_password():
            return secrets.token_hex(PASSWORD_BYTES)

    seen = set()
    lines = []
    for line in example_text.splitlines():
        match = ASSIGNMENT.match(line)
        if match and match.group("name") not in seen:
            seen.add(match.group("name"))
            line = match.group("head") + new_password()
        lines.append(line)
    missing = [name for name in PASSWORD_VARIABLES if name not in seen]
    if missing:
        raise EnvError("the example file does not set " + " and ".join(missing))
    return "\n".join(lines) + "\n"


def create(example_path, out_path):
    """Write ``out_path`` from ``example_path`` and return True; return False when it exists.

    The file is created exclusively (an ``.env`` that appears in between is not overwritten)
    and with mode 0600. Raises EnvError when the example cannot be read or is incomplete.
    """
    if os.path.lexists(out_path):
        return False
    try:
        with open(example_path, encoding="utf-8") as handle:
            example_text = handle.read()
    except OSError as exc:
        raise EnvError(f"cannot read {example_path}: {exc.strerror or exc}")
    text = render(example_text)
    try:
        descriptor = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    return True


def main(argv=None):
    """Run the command line; return the exit code (0 created or kept, 1 an EnvError)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--example", default=".env.example", help="the template (default: %(default)s)")
    parser.add_argument("--out", default=".env", help="the file to create (default: %(default)s)")
    args = parser.parse_args(argv)
    try:
        created = create(args.example, args.out)
    except EnvError as exc:
        print(f"make_env: {exc}", file=sys.stderr)
        return 1
    if created:
        print(
            f"Created {args.out} from {args.example} with two generated passwords "
            "(they are in that file, which git ignores)."
        )
    else:
        print(f"{args.out} exists: kept as it is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
