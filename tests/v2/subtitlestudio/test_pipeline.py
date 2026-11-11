from __future__ import annotations

import re
import sys
import types
from pathlib import Path

from tests.subtitlestudio_support.loader import load_domain

GEN = "v2"


def test_generation_uses_local_sidecar_and_skips_asr(tmp_path: Path, gen: str = GEN):
    models = load_domain(gen, "core.models")
    store_mod = load_domain(gen, "storage.job_store")
    gen_mod = load_domain(gen, "pipeline.generation")
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Movie.en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    asr_called = {"n": 0}

    def asr(*_args, **_kwargs):
        asr_called["n"] += 1
        return None

    def llm(prompt, **_kwargs):
        match = re.search(r"\[([^\]]+)\]", prompt)
        cue_id = match.group(1) if match else "1"
        return f'[{{"id":"{cue_id}","zh":"你好"}}]'

    pipeline = gen_mod.GenerationPipeline(
        store,
        lambda: {
            "translate_enabled": True,
            "translate_backend": "llm_only",
            "target_languages": ["zh-Hans"],
            "export_preset": "library_zh",
            "export_layouts": ["mono"],
            "export_formats": ["srt"],
            "overwrite_policy": "overwrite",
            "enable_asr": True,
            "effects_enabled": False,
        },
        asr=asr,
        llm=llm,
    )
    job = models.Job(job_id="j-local", title="Movie", path=str(video), trigger="manual")
    store.save_job(job)
    saved = pipeline.run(job)
    assert saved.status == "success"
    assert asr_called["n"] == 0
    written = tmp_path / "Movie.chi.zh-cn.srt"
    assert written.exists()
    assert "你好" in written.read_text(encoding="utf-8")


def test_online_source_precedes_local_sidecar(tmp_path: Path, gen: str = GEN):
    models = load_domain(gen, "core.models")
    store_mod = load_domain(gen, "storage.job_store")
    gen_mod = load_domain(gen, "pipeline.generation")
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Movie.en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nLocal\n", encoding="utf-8")
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    pipeline = gen_mod.GenerationPipeline(
        store,
        lambda: {"online_search_enabled": True, "enable_asr": True},
        searcher=lambda *_args: {"content": "1\n00:00:01,000 --> 00:00:02,000\nOnline\n", "filename": "online.srt", "lang": "en"},
        asr=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ASR should not run")),
    )
    job = models.Job(job_id="j-online", title="Movie", path=str(video))
    store.save_job(job)
    graph, asr_ran, _ = pipeline._resolve_source(job, {"online_search_enabled": True, "enable_asr": True}, "standard")
    assert graph.cues[0].text() == "Online"
    assert not asr_ran
    assert job.payload["subtitle_source"] == "online"


def test_online_wrong_metadata_full_bilingual_export_chain(tmp_path: Path, gen: str = GEN):
    models = load_domain(gen, "core.models")
    store_mod = load_domain(gen, "storage.job_store")
    gen_mod = load_domain(gen, "pipeline.generation")
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    llm_calls = []

    def llm(prompt, system="", role="translate"):
        llm_calls.append((prompt, system, role))
        match = re.search(r"\[([^\]]+)\]", prompt)
        cue_id = match.group(1) if match else "j-chain-1"
        return f'[{{"id":"{cue_id}","text":"这是一条英文字幕。"}}]'

    pipeline = gen_mod.GenerationPipeline(
        store,
        lambda: {
            "online_search_enabled": True,
            "timeline_auto_align": False,
            "translate_enabled": True,
            "translate_backend": "llm_only",
            "target_languages": ["zh-Hans", "en"],
            "export_preset": "library_zh",
            "export_layouts": ["stacked"],
            "export_formats": ["srt", "ass"],
            "overwrite_policy": "overwrite",
            "enable_asr": False,
            "effects_enabled": False,
        },
        searcher=lambda *_args, **_kwargs: {
            "content": "1\n00:00:01,000 --> 00:00:03,000\nThis is an English subtitle.\n",
            "filename": "wrongly-labelled.srt",
            "lang": "zh-Hans",
        },
        llm=llm,
    )
    job = models.Job(job_id="j-chain", title="Movie", path=str(video), trigger="manual")
    store.save_job(job)
    saved = pipeline.run(job)
    graph = store.get_graph(job.job_id)
    assert saved.status == "success"
    assert graph.source_lang == "en"
    assert graph.cues[0].texts == {
        "en": "This is an English subtitle.",
        "zh-Hans": "这是一条英文字幕。",
    }
    assert len(llm_calls) == 1
    outputs = [item for item in saved.payload["export"] if item.get("written")]
    assert {item["format"] for item in outputs} == {"srt", "ass"}
    for item in outputs:
        body = Path(item["path"]).read_text(encoding="utf-8")
        assert body.count("This is an English subtitle.") == 1
        assert body.count("这是一条英文字幕。") == 1


def test_partial_bilingual_source_keeps_mono_ass_in_primary_language(tmp_path: Path, gen: str = GEN):
    models = load_domain(gen, "core.models")
    store_mod = load_domain(gen, "storage.job_store")
    gen_mod = load_domain(gen, "pipeline.generation")
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    llm_calls = []

    def llm(prompt, system="", role="translate"):
        llm_calls.append((prompt, system, role))
        return "[]"

    content = (
        "1\n00:00:01,000 --> 00:00:03,000\n"
        "这是一句中文原文字幕。\nThis is its English translation.\n\n"
        "2\n00:00:03,200 --> 00:00:05,000\nNo.\n"
    )
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    pipeline = gen_mod.GenerationPipeline(
        store,
        lambda: {
            "online_search_enabled": True,
            "timeline_auto_align": False,
            "translate_enabled": True,
            "translate_backend": "llm_only",
            "target_languages": ["zh-Hans", "en"],
            "export_preset": "library_zh",
            "export_layouts": ["mono"],
            "export_formats": ["ass"],
            "overwrite_policy": "overwrite",
            "enable_asr": False,
            "effects_enabled": False,
        },
        searcher=lambda *_args, **_kwargs: {
            "content": content,
            "filename": "Movie.zh.srt",
            "lang": "zh-Hans",
        },
        llm=llm,
    )
    job = models.Job(job_id="partial-bilingual-ass", title="Movie", path=str(video), trigger="manual")
    store.save_job(job)

    saved = pipeline.run(job)

    graph = store.get_graph(job.job_id)
    assert saved.status == "success"
    assert graph.source_lang == "zh-Hans"
    assert graph.cues[0].texts == {
        "zh-Hans": "这是一句中文原文字幕。",
        "en": "This is its English translation.",
    }
    assert graph.cues[1].texts == {"en": "No."}
    assert llm_calls == []
    exported = [item for item in saved.payload["export"] if item.get("written")]
    assert len(exported) == 1
    ass = Path(exported[0]["path"]).read_text(encoding="utf-8")
    dialogue = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue) == 1
    assert dialogue[0].endswith("这是一句中文原文字幕。")
    assert "This is its English translation." not in ass
    assert "No." not in ass


def test_embedded_follows_failed_online_search(tmp_path: Path, monkeypatch, gen: str = GEN):
    models = load_domain(gen, "core.models")
    store_mod = load_domain(gen, "storage.job_store")
    gen_mod = load_domain(gen, "pipeline.generation")
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    calls = {"search": 0, "asr": 0}
    embedded = models.CueGraph(job_id="j", source_lang="en", cues=[
        models.Cue(cue_id="e1", index=1, start_ms=1200, end_ms=2000, texts={"source": "Hello."})
    ])
    monkeypatch.setattr(gen_mod, "extract_best_embedded", lambda *_args: embedded)
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    pipeline = gen_mod.GenerationPipeline(
        store, lambda: {"enable_asr": True},
        searcher=lambda *_args: calls.__setitem__("search", calls["search"] + 1),
        asr=lambda *_args, **_kwargs: calls.__setitem__("asr", calls["asr"] + 1),
    )
    job = models.Job(job_id="j", title="Movie", path=str(video))
    graph, asr_ran, _ = pipeline._resolve_source(job, {"enable_asr": True}, "search_then_translate")
    assert graph is embedded
    assert graph.cues[0].start_ms == 1200
    assert not asr_ran
    assert calls == {"search": 1, "asr": 0}


def test_rerun_skips_own_export_but_uses_original_sidecar(tmp_path: Path, gen: str = GEN):
    models = load_domain(gen, "core.models")
    store_mod = load_domain(gen, "storage.job_store")
    gen_mod = load_domain(gen, "pipeline.generation")
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    generated = tmp_path / "Movie.default.chi.zh-cn.srt"
    generated.write_text("1\n00:00:01,000 --> 00:00:02,000\nOld generated\n", encoding="utf-8")
    original = tmp_path / "Movie.en.srt"
    original.write_text("1\n00:00:03,000 --> 00:00:04,000\nOriginal\n", encoding="utf-8")
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    job = models.Job(job_id="j", title="Movie", path=str(video), payload={
        "export": [{"path": str(generated), "written": True}]
    })
    store.save_job(job)
    pipeline = gen_mod.GenerationPipeline(store, lambda: {})
    graph = pipeline._load_local_sidecar(job)
    assert graph.cues[0].text() == "Original"


def test_audio_alignment_applies_only_validated_matching_cues(tmp_path: Path, monkeypatch, gen: str = GEN):
    models = load_domain(gen, "core.models")
    timeline = load_domain(gen, "pipeline.timeline")
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    graph = models.CueGraph(job_id="j", source_lang="en", cues=[
        models.Cue(cue_id="a", index=1, start_ms=1000, end_ms=2000, texts={"en": "Hello"})
    ])
    parser = types.SimpleNamespace(parse_args=lambda args: args)
    fake_pkg = types.ModuleType("ffsubsync")
    fake_sub = types.ModuleType("ffsubsync.ffsubsync")
    fake_sub.make_parser = lambda: parser
    def run(args):
        Path(args[args.index("-o") + 1]).write_text(
            "1\n00:00:04,000 --> 00:00:05,000\nHello\n", encoding="utf-8"
        )
        return {"retval": 0, "sync_was_successful": True, "offset_seconds": 3.0}
    fake_pkg.run = run
    monkeypatch.setitem(sys.modules, "ffsubsync", fake_pkg)
    monkeypatch.setitem(sys.modules, "ffsubsync.ffsubsync", fake_sub)
    assert timeline.align_to_audio(graph, str(video), tmp_path, {}, None).cues[0].start_ms == 4000


def test_strm_never_calls_asr(tmp_path: Path, gen: str = GEN):
    models = load_domain(gen, "core.models")
    store_mod = load_domain(gen, "storage.job_store")
    gen_mod = load_domain(gen, "pipeline.generation")
    strm = tmp_path / "Show.strm"
    strm.write_text("http://example/video", encoding="utf-8")
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    asr_called = {"n": 0}
    pipeline = gen_mod.GenerationPipeline(
        store,
        lambda: {"enable_asr": True, "translate_enabled": False, "effects_enabled": False},
        asr=lambda *_a, **_k: asr_called.__setitem__("n", asr_called["n"] + 1) or None,
        searcher=lambda *_a, **_k: None,
    )
    job = models.Job(job_id="j-strm", title="Show", path=str(strm), trigger="strm", is_strm=True)
    store.save_job(job)
    saved = pipeline.run(job)
    assert asr_called["n"] == 0
    assert saved.status == "failed"


def test_generation_writes_step_logs(tmp_path: Path, gen: str = GEN):
    models = load_domain(gen, "core.models")
    store_mod = load_domain(gen, "storage.job_store")
    gen_mod = load_domain(gen, "pipeline.generation")
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Movie.en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    lines = []

    class Sink:
        def info(self, message, *args):
            lines.append(message % args if args else message)

        def warning(self, message, *args):
            lines.append(message % args if args else message)

        def error(self, message, *args):
            lines.append(message % args if args else message)

    def llm(prompt, **_kwargs):
        match = re.search(r"\[([^\]]+)\]", prompt)
        cue_id = match.group(1) if match else "1"
        return f'[{{"id":"{cue_id}","zh":"你好"}}]'

    pipeline = gen_mod.GenerationPipeline(
        store,
        lambda: {
            "translate_enabled": True,
            "translate_backend": "llm_only",
            "target_languages": ["zh-Hans"],
            "export_preset": "library_zh",
            "export_layouts": ["mono"],
            "export_formats": ["srt"],
            "overwrite_policy": "overwrite",
            "enable_asr": False,
            "effects_enabled": False,
        },
        llm=llm,
        logger=Sink(),
    )
    job = models.Job(job_id="j-log", title="Movie", path=str(video), trigger="manual")
    store.save_job(job)
    pipeline.run(job)
    text = "\n".join(lines)
    assert "[SubtitleStudio] [Step 0] 开始处理" in text
    assert "[Step 1] 查找本地外挂" in text
    assert "[Step 5] 翻译开始" in text
    assert "[Step 7] 写出导出包" in text
    assert "处理完成" in text


def test_generation_notifies_success_and_failure(tmp_path: Path, gen: str = GEN):
    models = load_domain(gen, "core.models")
    store_mod = load_domain(gen, "storage.job_store")
    gen_mod = load_domain(gen, "pipeline.generation")
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Movie.en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    notices = []

    def llm(prompt, **_kwargs):
        match = re.search(r"\[([^\]]+)\]", prompt)
        cue_id = match.group(1) if match else "1"
        return f'[{{"id":"{cue_id}","zh":"你好"}}]'

    pipeline = gen_mod.GenerationPipeline(
        store,
        lambda: {
            "send_notify": True,
            "notify_on": ["success", "failed"],
            "translate_enabled": True,
            "translate_backend": "llm_only",
            "target_languages": ["zh-Hans"],
            "export_preset": "library_zh",
            "export_layouts": ["mono"],
            "export_formats": ["srt"],
            "overwrite_policy": "overwrite",
            "enable_asr": False,
            "effects_enabled": False,
        },
        llm=llm,
        notify=notices.append,
    )
    job = models.Job(job_id="j-ok", title="Movie", path=str(video), trigger="manual")
    store.save_job(job)
    saved = pipeline.run(job)
    assert saved.status == "success"
    assert [item.status for item in notices] == ["success"]

    notices.clear()
    silent = gen_mod.GenerationPipeline(
        store,
        lambda: {"send_notify": False, "enable_asr": False, "translate_enabled": False, "effects_enabled": False},
        searcher=lambda *_a, **_k: None,
        notify=notices.append,
    )
    strm = tmp_path / "Show.strm"
    strm.write_text("http://example/video", encoding="utf-8")
    failed = models.Job(job_id="j-fail", title="Show", path=str(strm), trigger="strm", is_strm=True)
    store.save_job(failed)
    result = silent.run(failed)
    assert result.status == "failed"
    assert notices == []

    loud = gen_mod.GenerationPipeline(
        store,
        lambda: {"send_notify": True, "notify_on": ["failed"], "enable_asr": False, "translate_enabled": False, "effects_enabled": False},
        searcher=lambda *_a, **_k: None,
        notify=notices.append,
    )
    again = models.Job(job_id="j-fail-2", title="Show", path=str(strm), trigger="strm", is_strm=True)
    store.save_job(again)
    loud.run(again)
    assert [item.status for item in notices] == ["failed"]


def test_effects_character_brief_is_top_note(gen: str = GEN):
    models = load_domain(gen, "core.models")
    effects = load_domain(gen, "pipeline.effects")
    graph = models.CueGraph(
        job_id="j",
        cues=[models.Cue(cue_id="1", index=1, start_ms=0, end_ms=2000, texts={"zh-Hans": "钢铁侠来了"})],
    )
    built = effects.EffectsBuilder(
        {"effects_enabled": True, "effects_character_briefs": True, "effects_briefing": True},
        research=lambda *_a, **_k: "钢铁侠是很帅的男人",
    ).apply(graph, title="Iron Man")
    assert built.notes
    assert built.notes[0].kind == "note"
    assert "钢铁侠" in built.notes[0].text("note")
