"""Admin CLI to pre-provision users (no self-signup path exists in the app).

    python manage_users.py add <username>       # prompts for a password
    python manage_users.py list
    python manage_users.py remove <username>

`add` generates the user's api_key and prints it once. The user logs in with
name+password (POST /login) to retrieve that key; the key is then their
X-API-Key credential.
"""

import getpass
import sys
from datetime import datetime, timezone

import auth_utils
import persistence


def _add(username: str) -> None:
    if persistence.get_user(username):
        print(f"User {username!r} already exists. Use 'remove' first to reset.")
        sys.exit(1)
    pw = getpass.getpass(f"Password for {username}: ")
    pw2 = getpass.getpass("Confirm password: ")
    if pw != pw2:
        print("Passwords do not match.")
        sys.exit(1)
    if not pw:
        print("Password must not be empty.")
        sys.exit(1)
    salt, pw_hash = auth_utils.hash_password(pw)
    api_key = auth_utils.new_api_key()
    persistence.save_user(username, salt, pw_hash, api_key,
                          datetime.now(timezone.utc).isoformat())
    print(f"Created user {username!r}.")
    print(f"API key (share with the user, shown once): {api_key}")


def _list() -> None:
    users = persistence.list_users()
    if not users:
        print("(no users)")
        return
    for u in users:
        print(f"{u['username']}\t{u['created_at']}")


def _remove(username: str) -> None:
    if not persistence.get_user(username):
        print(f"User {username!r} not found.")
        sys.exit(1)
    persistence.delete_user(username)
    print(f"Removed user {username!r}.")


def main(argv: list) -> None:
    if len(argv) < 1:
        print(__doc__)
        sys.exit(2)
    cmd = argv[0]
    if cmd == "add" and len(argv) == 2:
        _add(argv[1])
    elif cmd == "list" and len(argv) == 1:
        _list()
    elif cmd == "remove" and len(argv) == 2:
        _remove(argv[1])
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
