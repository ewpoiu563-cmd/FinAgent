"""Conservative lexical policy for Phase 1 SELECT queries.

This is a policy check, not a full SQL grammar parser. SQLite checks syntax
and schema during execution. Comments are rejected even inside quoted text.
"""

import re


FORBIDDEN_KEYWORDS = frozenset(
    "INSERT UPDATE DELETE DROP ALTER CREATE REPLACE TRUNCATE ATTACH DETACH "
    "VACUUM PRAGMA".split()
)
_TOKEN = re.compile(
    r"'(?:(?:'')|[^'])*'|\"(?:(?:\"\")|[^\"])*\"|"
    r"`(?:(?:``)|[^`])*`|\[[^\]]*\]|[\w$]+|[^\s]",
    re.UNICODE,
)


def validate_sql(sql: str) -> dict:
    """Allow one SELECT, optionally terminated by one semicolon, without CTEs."""
    def reject(message):
        return {"valid": False, "error": message}

    if not isinstance(sql, str) or not sql.strip():
        return reject("SQL must be a non-empty string")
    if "\x00" in sql:
        return reject("NUL characters are not allowed")
    if any(marker in sql for marker in ("--", "/*", "*/")):
        return reject("SQL comments are not allowed")

    tokens = _TOKEN.findall(sql)
    if any(token in ("'", '"', "`", "[") for token in tokens):
        return reject("unterminated quoted string or identifier")
    if any(token.upper() == "WITH" for token in tokens):
        return reject("WITH / CTE queries are not supported")
    if tokens[0].upper() != "SELECT":
        return reject("only SELECT statements are allowed")
    if ";" in tokens[:-1] or tokens.count(";") > 1:
        return reject("only one SQL statement is allowed")
    for token in tokens:
        if token.upper() in FORBIDDEN_KEYWORDS:
            return reject(f"forbidden SQL keyword: {token.upper()}")
    return {"valid": True, "error": None}
