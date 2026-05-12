from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FORBIDDEN_DB_PACKAGES = {
    "mysqlclient",
    "pymysql",
    "psycopg2",
    "psycopg2-binary",
    "django",
    "sqlalchemy-django",
}


def _requirements_lines() -> list[str]:
    text = (_REPO_ROOT / "requirements.txt").read_text()
    return [line.strip().lower() for line in text.splitlines() if line.strip()]


def test_requirements_contains_no_db_drivers():
    names = {line.split("==", 1)[0].split("[", 1)[0] for line in _requirements_lines()}
    leaked = names & _FORBIDDEN_DB_PACKAGES
    assert not leaked, f"DB drivers leaked into requirements.txt: {sorted(leaked)}"


def test_dockerfile_exposes_only_port_8001():
    df = (_REPO_ROOT / "src" / "api" / "Dockerfile").read_text()
    exposes = [
        line.strip()
        for line in df.splitlines()
        if line.strip().upper().startswith("EXPOSE")
    ]
    assert exposes == ["EXPOSE 8001"], exposes


def test_source_tree_has_no_django_imports():
    for py in (_REPO_ROOT / "src").rglob("*.py"):
        text = py.read_text()
        for banned in ("import django", "from django"):
            assert banned not in text, f"{py}: contains '{banned}'"
