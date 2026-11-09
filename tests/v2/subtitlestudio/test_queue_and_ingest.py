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


def test_reclaim_orphan_running_unblocks_queue(tmp_path: Path, gen: str = GEN):
    models = load_domain(gen, "core.models")
    store_mod = load_domain(gen, "storage.job_store")
    sched_mod = load_domain(gen, "pipeline.scheduler")
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    stuck = models.Job(job_id="stuck", title="Stuck", path="/stuck.mkv", status="running", priority="P0")
    store.save_job(stuck)
    assert store.running_job() is not None
    scheduler = sched_mod.JobScheduler(store, lambda job: None)
    scheduler.start()
    scheduler.stop()
    stuck = store.get_job("stuck")
    assert stuck.status == "failed"
    assert "回收" in (stuck.error or "")
    assert store.running_job() is None


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


def test_scheduler_runs_jobs_together(tmp_path: Path, gen: str = GEN):
    import threading
    import time

    store_mod = load_domain(gen, "storage.job_store")
    sched_mod = load_domain(gen, "pipeline.scheduler")
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    started = []
    release = threading.Event()

    def runner(job):
        started.append(job.job_id)
        release.wait(2)

    scheduler = sched_mod.JobScheduler(store, runner)
    scheduler.config_getter = lambda: {"queue_workers": 2}
    scheduler.enqueue(title="A", path=str(tmp_path / "A.mkv"), trigger="manual", force=True)
    scheduler.enqueue(title="B", path=str(tmp_path / "B.mkv"), trigger="manual", force=True)
    scheduler._step()
    deadline = time.monotonic() + 1
    while len(started) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(started) == 2
    assert store.running_count() == 2
    release.set()


def test_scheduler_notifies_skipped_and_cancelled(tmp_path: Path, gen: str = GEN):
    store_mod = load_domain(gen, "storage.job_store")
    sched_mod = load_domain(gen, "pipeline.scheduler")
    gates = load_domain(gen, "ingest.gates")
    store = store_mod.JobStore(tmp_path, plugin_id="SubtitleStudio")
    notices = []
    video = tmp_path / "Dune.mkv"
    video.write_bytes(b"x" * 20 * 1024 * 1024)
    (tmp_path / "Dune.zh-Hans.srt").write_text("1", encoding="utf-8")
    config = {
        "send_notify": True,
        "notify_on": ["skipped", "cancelled"],
        "skip_existing_chinese": True,
        "min_file_mb": 10,
    }
    scheduler = sched_mod.JobScheduler(store, lambda job: None, notify=notices.append)
    scheduler.config_getter = lambda: config
    skipped = scheduler.enqueue(title="Dune", path=str(video), trigger="event", config=config)
    assert skipped.status == "skipped"
    assert "中字" in (skipped.error or "")
    assert [item.status for item in notices] == ["skipped"]
    ok, reason = gates.evaluate_gates(scheduler.config_getter(), str(video), {})
    assert ok is False and "中字" in reason

    notices.clear()
    pending = scheduler.enqueue(title="Other", path=str(tmp_path / "Other.mkv"), trigger="manual", force=True)
    cancelled = scheduler.cancel(pending.job_id)
    assert cancelled.status == "cancelled"
    assert [item.status for item in notices] == ["cancelled"]


def test_offpeak_window_cross_midnight(gen: str = GEN):
    sched = load_domain(gen, "pipeline.scheduler")
    window = sched.parse_offpeak_window("22:00-06:00")
    from datetime import time
    assert sched.in_offpeak_window(window, time(23, 0)) is True
    assert sched.in_offpeak_window(window, time(8, 0)) is False


def test_search_html_parser_ignores_search_links(gen: str = GEN):
    online = load_domain(gen, "providers.online")
    html = (
        '<a href="/search?q=dune">热门</a>'
        '<a href="/d/123">沙丘 特效</a>'
        '<a href="/d/456">沙丘 中英双语.ass</a>'
    )
    rows = online.parse_search_html(html, "subhd", "https://subhd.tv")
    assert len(rows) == 2
    assert rows[0]["title"] == "沙丘 特效"
    assert rows[0]["lang"] == "source"
    assert rows[1]["lang"] == "bilingual"

    service = online.OnlineSearchService(
        lambda *_args, **_kwargs: {"sub": {"subs": [
            {"native_name": "Movie.ass", "url": "https://example/1"},
            {"native_name": "Movie 中英双语.ass", "url": "https://example/2"},
            {"native_name": "Movie.en.srt", "url": "https://example/3", "lang": "eng"},
        ]}},
        {},
    )
    assrt = service._assrt("Movie", {"assrt_api_key": "token"})
    assert [item["lang"] for item in assrt] == ["source", "bilingual", "en"]


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


def test_wrong_language_metadata_migrates_and_enters_translation(gen: str = GEN):
    trans = load_domain(gen, "pipeline.translation")
    models = load_domain(gen, "core.models")
    packager = load_domain(gen, "packager.export_pack")
    calls = []

    def llm(prompt, system="", role="translate"):
        calls.append({"prompt": prompt, "system": system, "role": role})
        return '[{"id":"1","text":"这是一条英文字幕。"}]'

    graph = models.CueGraph(
        job_id="wrong-language",
        source_lang="zh-Hans",
        cues=[models.Cue(
            cue_id="1",
            index=1,
            start_ms=0,
            end_ms=1000,
            texts={"zh-Hans": "This is an English subtitle."},
        )],
    )
    service = trans.TranslationService(
        {"translate_enabled": True, "translate_backend": "llm_only"},
        http=lambda *a, **k: None,
        llm=llm,
    )
    service.translate_graph(graph, ["zh-Hans", "en"])
    assert graph.source_lang == "en"
    assert graph.cues[0].texts == {
        "en": "This is an English subtitle.",
        "zh-Hans": "这是一条英文字幕。",
    }
    assert len(calls) == 1
    stacked = packager.render_plan_text(graph, {
        "kind": "dialogue",
        "format": "srt",
        "layout": "stacked",
        "langs": ["zh-Hans", "en"],
        "stack": "main_bottom",
    })
    lines = [line for line in stacked.splitlines() if line and "-->" not in line and not line.isdigit()]
    assert lines == ["This is an English subtitle.", "这是一条英文字幕。"]


def test_free_translation_rejects_original_and_tries_next_engine(gen: str = GEN):
    engines = load_domain(gen, "pipeline.mt_engines")
    original = "This is an English subtitle."
    edge_payload = [{"translations": [{"text": original}]}]
    gtx_payload = [[[original, original, None, None]]]
    assert engines._pick_translated(edge_payload, original) is None
    assert engines._pick_gtx(gtx_payload, original) is None
    assert engines._pick_translated([{"translations": [{"text": "NASA"}]}], "NASA") == "NASA"

    calls = []

    def http(method, url, **_kwargs):
        calls.append(url)
        if "edge.microsoft.com" in url:
            return edge_payload
        return [[[
            "这是一条英文字幕。",
            original,
            None,
            None,
        ]]]

    router = engines.FreeMtRouter(http, {"mt_engines": ["edge", "gtx"]})
    assert router.translate_batch([original], source="en", target="zh-Hans") == ["这是一条英文字幕。"]
    assert len(calls) == 2

    batch = [f"English subtitle sentence {index}." for index in range(6)]

    def partial_http(_method, url, **kwargs):
        if "edge.microsoft.com" in url:
            source_text = kwargs["json_body"][0]["Text"]
            translated = "第一句。" if source_text == batch[0] else source_text
            return [{"translations": [{"text": translated}]}]
        return [[["后续译文。", "source", None, None]]]

    partial_router = engines.FreeMtRouter(partial_http, {
        "mt_engines": ["edge", "gtx"],
        "parallel_workers": 1,
        "mt_max_consecutive_errors": 2,
    })
    batch_result = partial_router.translate_batch(batch, source="en", target="zh-Hans")
    assert batch_result[0] == "第一句。"
    assert batch_result[1:] == ["后续译文。"] * 5

    def configured_http(_method, url, **_kwargs):
        if url.startswith("https://deeplx.example"):
            return {"data": original}
        return {"translatedText": "这是一条英文字幕。"}

    configured_router = engines.FreeMtRouter(configured_http, {
        "mt_engines": ["deeplx", "libre"],
        "deeplx_url": "https://deeplx.example",
        "libre_url": "https://libre.example",
    })
    assert configured_router.translate_batch([original], source="en", target="zh-Hans") == ["这是一条英文字幕。"]


def test_invalid_free_and_llm_outputs_never_fill_target_key(gen: str = GEN):
    trans = load_domain(gen, "pipeline.translation")
    models = load_domain(gen, "core.models")
    original = "This is an English subtitle."
    llm_calls = []

    def http(_method, url, **_kwargs):
        if "edge.microsoft.com" in url:
            return [{"translations": [{"text": original}]}]
        return [[[original, original, None, None]]]

    def llm(_prompt, **_kwargs):
        llm_calls.append(1)
        return f'[{{"id":"1","text":"{original}"}}]'

    graph = models.CueGraph(
        job_id="invalid-output",
        source_lang="en",
        cues=[models.Cue(cue_id="1", index=1, start_ms=0, end_ms=1000, texts={"en": original})],
    )
    service = trans.TranslationService(
        {"translate_enabled": True, "translate_backend": "free_first", "mt_engines": ["edge", "gtx"]},
        http=http,
        llm=llm,
    )
    service.translate_graph(graph, ["zh-Hans"])
    assert llm_calls == [1]
    assert "zh-Hans" not in graph.cues[0].texts
    assert any(issue.code == "translate_failed" for issue in graph.cues[0].issues)


def test_llm_prompt_names_source_and_each_target_language(gen: str = GEN):
    trans = load_domain(gen, "pipeline.translation")
    models = load_domain(gen, "core.models")
    cases = [
        ("en", "This is an English subtitle.", "zh-Hans", "这是一条英文字幕。"),
        ("zh-Hans", "你好世界，欢迎观看。", "en", "Hello world, welcome."),
        ("en", "This is an English subtitle.", "zh-Hant", "這是一條英文字幕。"),
        ("en", "This is an English subtitle.", "ja", "これは英語の字幕です。"),
        ("en", "This is an English subtitle.", "ko", "이것은 영어 자막입니다."),
    ]
    for source_lang, source_text, target_lang, translated_text in cases:
        calls = []

        def llm(prompt, system="", role="translate"):
            calls.append((prompt, system, role))
            return f'[{{"id":"1","text":"{translated_text}"}}]'

        graph = models.CueGraph(
            job_id=f"{source_lang}-{target_lang}",
            source_lang=source_lang,
            cues=[models.Cue(cue_id="1", index=1, start_ms=0, end_ms=1000, texts={source_lang: source_text})],
        )
        service = trans.TranslationService(
            {"translate_enabled": True, "translate_backend": "llm_only"},
            http=lambda *a, **k: None,
            llm=llm,
        )
        service.translate_graph(graph, [target_lang])
        assert graph.cues[0].text(target_lang) == translated_text
        assert len(calls) == 1
        prompt, system, role = calls[0]
        assert source_lang in prompt or source_lang in system
        assert target_lang in prompt and target_lang in system
        assert '"text"' in system
        assert role == "translate"
