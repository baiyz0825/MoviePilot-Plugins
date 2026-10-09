from __future__ import annotations

from pathlib import Path

from tests.subtitlestudio_support.loader import load_domain

GEN = "v2"


def test_multiline_paths_use_real_newlines_only(gen: str = GEN):
    paths = load_domain(gen, "core.paths")
    assert paths.parse_multiline_paths("/a\n/b\n\n/a") == ["/a", "/b"]
    # 字面 \n 两个字符不能当分隔，避免海拉鲁那种框里画出 \n
    assert paths.parse_multiline_paths("/media/movies\\n/media/tv") == ["/media/movies\\n/media/tv"]


def test_identity_pair_and_v2_fallback(gen: str = GEN):
    identity = load_domain(gen, "core.identity")
    assert identity.build_media_key("themoviedb", "693134") == "themoviedb:693134"
    assert identity.build_media_key("TMDB", "0") == ""
    v2 = identity.identity_from_v2_ids("693134", None)
    assert v2["media_source"] == "themoviedb"
    assert identity.debounce_key(v2, "/media/a.mkv").startswith("themoviedb:693134")


def test_config_defaults_and_language_limit(gen: str = GEN):
    schema = load_domain(gen, "core.config_schema")
    cfg = schema.normalize_plugin_config({
        "enabled": "true",
        "target_languages": ["zh-Hans", "en", "ja", "ko"],
        "watch_paths": "/m1\n/m2\n",
        "ingest_on_watch": True,
        "export_preset": "plex",
    })
    assert cfg["enabled"] is True
    assert cfg["ingest_on_event"] is True
    assert cfg["ingest_on_watch"] is True
    assert cfg["target_languages"] == ["zh-Hans", "en", "ja"]
    assert cfg["watch_paths"] == "/m1\n/m2"
    assert cfg["mark_default"] is False
    assert cfg["format_repair_enabled"] is True
    assert cfg["abort_on_high_failure"] is False
    keys = {item["key"] for item in schema.FIELDS}
    assert "watch_paths" in keys and "openai_endpoints" in keys
    assert "ass_style" in keys
    assert "notify_on" in keys
    assert cfg["send_notify"] is False
    assert cfg["notify_on"] == ["success", "failed"]
    assert cfg["ass_style"]["sizes"] == [22, 17, 14]


def test_notify_payload_and_gating(gen: str = GEN):
    notify = load_domain(gen, "core.notify")
    models = load_domain(gen, "core.models")
    success = models.Job(
        job_id="n1",
        title="沙丘",
        path="/media/Dune.mkv",
        status="success",
        trigger="event",
        payload={
            "poster": "http://poster/dune.jpg",
            "export": [
                {"written": True, "filename": "Dune.default.chi.zh-cn.srt"},
                {"written": False, "filename": "Dune.en.srt"},
            ],
        },
    )
    failed = models.Job(job_id="n2", title="沙丘", path="/media/Dune.mkv", status="failed", error="没有可用字幕源")
    assert notify.should_notify({"send_notify": False}, success) is False
    assert notify.should_notify({"send_notify": True}, success) is True
    assert notify.should_notify({"send_notify": True, "notify_on": ["failed"]}, success) is False
    assert notify.should_notify({"send_notify": True, "notify_on": ["failed"]}, failed) is True
    payload = notify.build_notify_payload(success)
    assert payload["title"] == "字幕工坊 · 完成"
    assert "Dune.default.chi.zh-cn.srt" in payload["text"]
    assert "来源：整理入库" in payload["text"]
    assert payload["image"] == "http://poster/dune.jpg"
    fail_payload = notify.build_notify_payload(failed)
    assert fail_payload["title"] == "字幕工坊 · 失败"
    assert "没有可用字幕源" in fail_payload["text"]


def test_cuegraph_roundtrip_and_ass_styles(gen: str = GEN):
    cuegraph = load_domain(gen, "core.cuegraph")
    srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:03,200 --> 00:00:05,000\nWorld\n"
    graph = cuegraph.parse_srt(srt, lang="en", job_id="j1")
    assert len(graph.cues) == 2
    graph.cues[0].texts["zh-Hans"] = "你好"
    ass = cuegraph.render_ass(graph, ["zh-Hans", "en"], sizes=[22, 17])
    assert "Style: Lang1" in ass and "Style: Lang2" in ass
    assert "你好" in ass
    styled = cuegraph.render_ass(
        graph,
        ["zh-Hans"],
        style={"font_name": "Microsoft YaHei", "primary_color": "#FFE566", "sizes": [28, 18, 12], "bold": True},
    )
    assert "Microsoft YaHei" in styled
    assert ",28," in styled
    assert "&H0066E5FF" in styled
    assert styled.split("Style: Lang1,")[1].split(",")[6] == "-1"


def test_format_repair_keeps_good_sentences(gen: str = GEN):
    repair = load_domain(gen, "core.format_repair")
    raw = "好的，如下所示：\n```json\n[{\"id\":\"a\",\"zh\":\"甲\"},{\"id\":\"c\",\"zh\":\"丙\",}]\n```"
    mapped, missing, steps = repair.repair_model_batch(raw, ["a", "b", "c"])
    assert mapped["a"] == "甲"
    assert mapped["c"] == "丙"
    assert missing == ["b"]
    assert "extract_json" in steps


def test_asr_repair_drops_empty_and_fixes_overlap(gen: str = GEN):
    models = load_domain(gen, "core.models")
    repair = load_domain(gen, "core.format_repair")
    graph = models.CueGraph(
        job_id="j",
        cues=[
            models.Cue(cue_id="1", index=1, start_ms=1000, end_ms=800, texts={"en": "ok"}),
            models.Cue(cue_id="2", index=2, start_ms=700, end_ms=1500, texts={"en": "嗯"}),
            models.Cue(cue_id="3", index=3, start_ms=1600, end_ms=1800, texts={"en": "fine"}),
        ],
    )
    repaired = repair.repair_asr_graph(graph)
    texts = [item.text() for item in repaired.cues]
    assert "嗯" not in texts
    assert repaired.cues[0].end_ms >= repaired.cues[0].start_ms


def test_export_plan_cartesian_and_extras(gen: str = GEN):
    naming = load_domain(gen, "core.naming")
    files = naming.export_plan({
        "export_preset": "library_zh",
        "target_languages": ["zh-Hans", "en"],
        "export_layouts": ["mono", "stacked"],
        "export_formats": ["srt"],
        "mark_default": True,
        "lang_stack": "main_bottom",
    }, "Movie")
    names = [item["filename"] for item in files]
    assert "Movie.default.chi.zh-cn.srt" in names
    assert any("bilingual" in name or "chi&eng" in name for name in names)
    extras = naming.extra_tracks({"effects_enabled": True, "enable_sdh": True, "save_asr_track": True, "target_languages": ["zh-Hans"]}, "Movie", asr_ran=True)
    kinds = {item["kind"] for item in extras}
    assert kinds == {"notes", "sdh", "asr"}
    assert naming.extra_tracks({"save_asr_track": True}, "Movie", asr_ran=False) == []


def test_export_plan_uses_player_language_tags(gen: str = GEN):
    naming = load_domain(gen, "core.naming")
    library = naming.export_plan({
        "export_preset": "library_zh",
        "target_languages": ["zh-Hans"],
        "export_layouts": ["mono"],
        "export_formats": ["srt"],
        "mark_default": True,
    }, "Movie")
    assert library[0]["filename"] == "Movie.default.chi.zh-cn.srt"
    plex = naming.export_plan({
        "export_preset": "plex",
        "target_languages": ["zh-Hans"],
        "export_layouts": ["mono"],
        "export_formats": ["srt"],
        "mark_default": True,
    }, "Movie")
    assert plex[0]["filename"] == "Movie.chi.srt"
    fnos = naming.export_plan({
        "export_preset": "fnos",
        "target_languages": ["zh-Hans"],
        "export_layouts": ["mono"],
        "export_formats": ["srt"],
        "mark_default": True,
    }, "Movie")
    assert fnos[0]["filename"] == "Movie.chs.srt"
    web = naming.export_plan({
        "export_preset": "web",
        "target_languages": ["zh-Hans"],
        "export_layouts": ["mono"],
        "export_formats": ["srt"],
        "mark_default": True,
    }, "Movie")
    assert web[0]["filename"] == "Movie.zh-CN.default.srt"
    sized = naming.export_plan({
        "export_preset": "library_zh",
        "target_languages": ["zh-Hans", "en"],
        "export_layouts": ["stacked"],
        "export_formats": ["ass"],
        "ass_style": {"sizes": [30, 16, 12]},
    }, "Movie")
    assert sized[0]["sizes"] == [30, 16]


def test_packager_writes_and_skip_policy(tmp_path: Path, gen: str = GEN):
    models = load_domain(gen, "core.models")
    packager = load_domain(gen, "packager.export_pack")
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    existing = tmp_path / "Movie.chi.zh-cn.srt"
    existing.write_text("old", encoding="utf-8")
    graph = models.CueGraph(
        job_id="j",
        cues=[models.Cue(cue_id="1", index=1, start_ms=0, end_ms=1000, texts={"zh-Hans": "你好"})],
    )
    skipped = packager.write_export_pack(
        {
            "export_preset": "library_zh",
            "target_languages": ["zh-Hans"],
            "export_layouts": ["mono"],
            "export_formats": ["srt"],
            "mark_default": False,
            "overwrite_policy": "skip",
            "encoding": "utf-8",
        },
        str(video),
        graph,
    )
    assert skipped[0]["written"] is False
    assert existing.read_text(encoding="utf-8") == "old"
    written = packager.write_export_pack(
        {
            "export_preset": "library_zh",
            "target_languages": ["zh-Hans"],
            "export_layouts": ["mono"],
            "export_formats": ["srt"],
            "mark_default": False,
            "overwrite_policy": "overwrite",
            "encoding": "utf-8",
        },
        str(video),
        graph,
    )
    assert written[0]["written"] is True
    assert "你好" in existing.read_text(encoding="utf-8")
