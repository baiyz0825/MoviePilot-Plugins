from __future__ import annotations

from tests.subtitlestudio_support.loader import load_domain

GEN = "v2"


def test_history_expands_fileitem_and_files_json(gen: str = GEN):
    history = load_domain(gen, "core.history")
    rows = history.expand_history_rows([
        {
            "title": "沙丘",
            "year": "2021",
            "type": "电影",
            "tmdbid": "438631",
            "dest": "/media/movies/Dune/Dune.mkv",
            "dest_fileitem": {"storage": "local", "path": "/media/movies/Dune/Dune.mkv", "name": "Dune.mkv"},
            "files": '["/media/movies/Dune/Dune.mkv","/media/movies/Dune/Dune-extra.mkv"]',
            "image": "https://image/dune.jpg",
            "status": True,
        },
        {
            "title": "人生切割术",
            "type": "电视剧",
            "tmdbid": "95396",
            "seasons": "S01",
            "episodes": "E02",
            "dest": "/media/tv/Severance/S01E02.mkv",
            "status": True,
        },
    ])
    paths = [item["path"] for item in rows]
    assert "/media/movies/Dune/Dune.mkv" in paths
    assert "/media/movies/Dune/Dune-extra.mkv" in paths
    dune = next(item for item in rows if item["path"].endswith("Dune.mkv"))
    assert dune["type"] == "movie"
    assert dune["media_source"] == "themoviedb"
    assert dune["origin"] == "transfer_history"
    show = next(item for item in rows if "Severance" in item["path"])
    assert show["type"] == "tv"
    assert show["season"] == 1
    assert show["episode"] == 2


def test_catalog_groups_history_for_manual_submit(tmp_path, gen: str = GEN):
    catalog_mod = load_domain(gen, "providers.catalog")
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")

    def history_loader(limit=800):
        return [{
            "title": "沙丘",
            "type": "movie",
            "tmdbid": "1",
            "dest": str(video),
            "status": True,
        }]

    catalog = catalog_mod.MediaCatalog(lambda: {"trust_transfer_history": True}, history_loader=history_loader)
    groups = catalog.list_groups(force=True)
    assert len(groups) == 1
    assert groups[0]["title"] == "沙丘"
    assert groups[0]["file_count"] == 1
    assert groups[0]["files"][0]["path"] == str(video)
