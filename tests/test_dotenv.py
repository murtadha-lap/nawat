"""A .env beside the project is part of the environment.

The researcher writes credentials into .env once; activating a venv or opening a
new shell should not silently drop them. Real environment variables still win,
so an explicit export overrides for one command.
"""

from __future__ import annotations

import pytest

from nawat import dotenv
from nawat.config import Config
from nawat.errors import NotFound


def test_values_parse_with_export_quotes_and_comments():
    values = dotenv.parse(
        """
        # a comment
        NAWAT_S3_BUCKET=ai-model
        export NAWAT_S3_ENDPOINT=http://192.168.0.155:9000
        NAWAT_CACHE_CEILING = 120GB
        QUOTED="  spaced  "
        SINGLE='literal $NOT_EXPANDED'
        TRAILING=value   # explained here
        ESCAPED="line\\nbreak"

        NOT A VARIABLE
        """
    )
    assert values["NAWAT_S3_BUCKET"] == "ai-model"
    assert values["NAWAT_S3_ENDPOINT"] == "http://192.168.0.155:9000"
    assert values["NAWAT_CACHE_CEILING"] == "120GB"
    assert values["QUOTED"] == "  spaced  "
    assert values["SINGLE"] == "literal $NOT_EXPANDED"
    assert values["TRAILING"] == "value"
    assert values["ESCAPED"] == "line\nbreak"
    assert "NOT A VARIABLE" not in values


def test_a_hash_inside_a_secret_survives():
    """Passwords contain # more often than lines contain inline comments."""
    values = dotenv.parse("NAWAT_S3_SECRET_KEY=pa#ssw0rd\nOTHER='has # inside'\n")
    assert values["NAWAT_S3_SECRET_KEY"] == "pa#ssw0rd"
    assert values["OTHER"] == "has # inside"


def test_the_nearest_env_file_is_found_from_a_subdirectory(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("NAWAT_S3_BUCKET=found-above\n")
    deep = tmp_path / "notebooks" / "ocr"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)

    assert dotenv.find() == tmp_path / ".env"


def test_no_env_file_anywhere_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dotenv, "MAX_DEPTH", 1)
    assert dotenv.find() is None


# -- how Config uses it ------------------------------------------------------


def test_config_reads_the_env_file_beside_the_project(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "NAWAT_S3_ENDPOINT=http://192.168.0.155:9000\n"
        "NAWAT_S3_BUCKET=ai-model\n"
        "NAWAT_S3_ACCESS_KEY=home\n"
        "NAWAT_CACHE_CEILING=120GB\n"
    )
    monkeypatch.chdir(tmp_path)

    config = Config.from_env({})

    assert config.endpoint == "http://192.168.0.155:9000"
    assert config.bucket == "ai-model"
    assert config.cache_ceiling == 120 * 10**9
    assert config.env_file == tmp_path / ".env"


def test_the_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("NAWAT_S3_BUCKET=from-file\n")
    monkeypatch.chdir(tmp_path)

    config = Config.from_env({"NAWAT_S3_BUCKET": "from-the-shell"})

    assert config.bucket == "from-the-shell"


def test_an_explicit_env_file_is_used(tmp_path):
    elsewhere = tmp_path / "staging.env"
    elsewhere.write_text("NAWAT_S3_BUCKET=staging\n")

    config = Config.from_env({}, env_file=elsewhere)

    assert config.bucket == "staging"
    assert config.env_file == elsewhere


def test_an_explicit_env_file_that_is_missing_says_so(tmp_path):
    with pytest.raises(NotFound, match="does not exist"):
        Config.from_env({}, env_file=tmp_path / "absent.env")


def test_env_file_can_be_named_by_variable(tmp_path):
    named = tmp_path / "prod.env"
    named.write_text("NAWAT_S3_BUCKET=named\n")

    config = Config.from_env({"NAWAT_ENV_FILE": str(named)})

    assert config.bucket == "named"


def test_an_empty_env_file_variable_means_use_none(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("NAWAT_S3_BUCKET=should-be-ignored\n")
    monkeypatch.chdir(tmp_path)

    config = Config.from_env({"NAWAT_ENV_FILE": ""})

    assert config.bucket == "nawat"  # the default
    assert config.env_file is None


def test_a_bad_env_file_variable_says_how_to_fix_it(tmp_path):
    with pytest.raises(NotFound, match="NAWAT_ENV_FILE"):
        Config.from_env({"NAWAT_ENV_FILE": str(tmp_path / "typo.env")})


def test_credentials_from_the_file_are_still_redacted(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("NAWAT_S3_SECRET_KEY=f1805acc-55a7-4964-849a-8076b3e1cfdf\n")
    monkeypatch.chdir(tmp_path)

    redacted = Config.from_env({}).redacted()

    assert redacted["secret_key"] == "set"
    assert "f1805acc" not in str(redacted)
