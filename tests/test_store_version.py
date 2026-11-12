import sqlite3

import pytest

from secom_quality.inference import ApplicationError
from secom_quality.storage import Store


def test_store_does_not_downgrade_a_newer_database(tmp_path):
    path = tmp_path / "runtime/application.sqlite3"
    path.parent.mkdir()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 2")
    connection.close()
    with pytest.raises(ApplicationError, match="schema version"):
        Store(tmp_path)
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    connection.close()
