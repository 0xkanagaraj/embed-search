def test_add_list_delete_file(isolated_db):
    db = isolated_db
    fid = db.add_file("a.txt", "/tmp/a.txt", 0)
    db.update_file_chunks(fid, 7)

    files = db.list_files()
    assert len(files) == 1
    assert files[0]["id"] == fid
    assert files[0]["n_chunks"] == 7

    db.delete_file(fid)
    assert db.list_files() == []


def test_reupload_same_filename_keeps_id(isolated_db):
    db = isolated_db
    first = db.add_file("a.txt", "/tmp/a.txt", 3)
    second = db.add_file("a.txt", "/tmp/new/a.txt", 5)
    assert first == second
    files = db.list_files()
    assert len(files) == 1
    assert files[0]["path"] == "/tmp/new/a.txt"
    assert files[0]["n_chunks"] == 5


def test_query_log_and_stats(isolated_db):
    db = isolated_db
    db.log_query("rag", 20, 5, 120.5, 800.2)
    db.log_query("search", 10, 3, 90.0, 95.0)
    db.log_query("rag", 15, 4, 100.0, 700.0)

    stats = db.stats()
    assert stats["n_queries"] == 3
    assert stats["by_mode"] == {"rag": 2, "search": 1}
    assert stats["avg_hits"] == round((5 + 3 + 4) / 3, 2)


def test_stats_when_empty(isolated_db):
    stats = isolated_db.stats()
    assert stats["n_queries"] == 0
    assert stats["by_mode"] == {}
    assert stats["n_files"] == 0
