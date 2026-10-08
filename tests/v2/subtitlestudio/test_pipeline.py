from __future__ import annotations

import re
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
    written = tmp_path / "Movie.zh-Hans.srt"
    assert written.exists()
    assert "你好" in written.read_text(encoding="utf-8")


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
    assert "[Step 4] 翻译开始" in text
    assert "[Step 6] 写出导出包" in text
    assert "处理完成" in text


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
