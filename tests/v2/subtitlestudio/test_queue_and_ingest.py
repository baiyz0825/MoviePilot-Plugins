from __future__ import annotations

from pathlib import Path

from tests.subtitlestudio_support.loader import load_domain

GEN = "v2"


def test_debounce_blocks_same_identity_within_five_minutes(gen: str = GEN):
    debounce = load_domain(gen, "ingest.debounce")
    box = debounce.IngestDebouncer(window_seconds=300)
    identity = {"media_source": "themoviedb", "media_id": "1"}
    assert box.allow(identity, "/a.mkv", now=1000) is True
    assert box.allow(identity, "/a.mkv", now=1100) is False
    assert box.allow(identity, "/a.mkv", now=1400) is True


def test_gates_skip_existing_chinese(tmp_path: Path, gen: str = GEN):
    gates = load_domain(gen, "ingest.gates")
    video = tmp_path / "Dune.mkv"
    video.write_bytes(b"x" * 20 * 1024 * 1024)
    (tmp_path / "Dune.zh-Hans.srt").write_text("1", encoding="utf-8")
    ok, reason = gates.evaluate_gates({"skip_existing_chinese": True, "min_file_mb": 10}, str(video), {})
    assert ok is False
    assert "中字" in reason


def test_job_store_cut_in_does_not_touch_running(tmp_path: Path, gen: str = GEN):
    store_mod = load_domain(gen, "storage.job_store")
    sched_mod = load_domain(gen, "pipeline.scheduler")
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    ran = []

    def runner(job):
        ran.append(job.job_id)

    scheduler = sched_mod.JobScheduler(store, runner)
    first = scheduler.enqueue(title="A", path="/a.mkv", trigger="manual", priority="P1", force=True)
    second = scheduler.enqueue(title="B", path="/b.mkv", trigger="manual", priority="P2", force=True)
    first.status = "running"
    store.save_job(first)
    scheduler.cut_in(second.job_id)
    second = store.get_job(second.job_id)
    assert second.priority == "P0"
    assert second.queue_rank < first.queue_rank or second.status == "pending"
    running = store.running_job()
    assert running and running.job_id == first.job_id


def test_offpeak_window_cross_midnight(gen: str = GEN):
    sched = load_domain(gen, "pipeline.scheduler")
    window = sched.parse_offpeak_window("22:00-06:00")
    from datetime import time
    assert sched.in_offpeak_window(window, time(23, 0)) is True
    assert sched.in_offpeak_window(window, time(8, 0)) is False


def test_search_html_parser_ignores_search_links(gen: str = GEN):
    online = load_domain(gen, "providers.online")
    html = '<a href="/search?q=dune">热门</a><a href="/d/123">沙丘 特效</a>'
    rows = online.parse_search_html(html, "subhd", "https://subhd.tv")
    assert len(rows) == 1
    assert rows[0]["title"] == "沙丘 特效"


def test_translation_partial_accept_and_no_placeholder(gen: str = GEN):
    trans = load_domain(gen, "pipeline.translation")
    models = load_domain(gen, "core.models")

    def llm(prompt, system="", role="translate"):
        return '[{"id":"1","zh":"你好"}]'

    service = trans.TranslationService(
        {"translate_enabled": True, "translate_backend": "llm_only", "format_repair_enabled": True, "write_failure_placeholder": False, "enable_batch": True, "batch_size": 20, "context_window": 5},
        http=lambda *a, **k: None,
        llm=llm,
    )
    graph = models.CueGraph(
        job_id="j",
        source_lang="en",
        cues=[
            models.Cue(cue_id="1", index=1, start_ms=0, end_ms=1000, texts={"en": "Hello"}),
            models.Cue(cue_id="2", index=2, start_ms=1000, end_ms=2000, texts={"en": "World"}),
        ],
    )
    service.translate_graph(graph, ["zh-Hans"])
    assert graph.cues[0].text("zh-Hans") == "你好"
    assert "[翻译失败]" not in graph.cues[1].text("zh-Hans")
    assert any(issue.code == "translate_failed" for issue in graph.cues[1].issues)
