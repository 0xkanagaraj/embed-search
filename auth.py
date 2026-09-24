import bcrypt
from db import create_user, get_user


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())


def signup(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if get_user(username):
        return False, "Username already exists."
    return create_user(username, hash_password(password)), "Account created."


def login(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    user = get_user(username)
    if not user or not verify_password(password, user["pwd_hash"]):
        return False, "Invalid credentials."
    return True, username