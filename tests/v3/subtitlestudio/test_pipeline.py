from tests.v2.subtitlestudio import test_pipeline as shared


def test_generation_uses_local_sidecar_and_skips_asr(tmp_path):
    shared.test_generation_uses_local_sidecar_and_skips_asr(tmp_path, gen="v3")


def test_embedded_source_precedes_online_and_asr(tmp_path, monkeypatch):
    shared.test_embedded_source_precedes_online_and_asr(tmp_path, monkeypatch, gen="v3")


def test_rerun_skips_own_export_but_uses_original_sidecar(tmp_path):
    shared.test_rerun_skips_own_export_but_uses_original_sidecar(tmp_path, gen="v3")


def test_audio_alignment_applies_only_validated_matching_cues(tmp_path, monkeypatch):
    shared.test_audio_alignment_applies_only_validated_matching_cues(tmp_path, monkeypatch, gen="v3")


def test_strm_never_calls_asr(tmp_path):
    shared.test_strm_never_calls_asr(tmp_path, gen="v3")


def test_generation_writes_step_logs(tmp_path):
    shared.test_generation_writes_step_logs(tmp_path, gen="v3")


def test_generation_notifies_success_and_failure(tmp_path):
    shared.test_generation_notifies_success_and_failure(tmp_path, gen="v3")


def test_effects_character_brief_is_top_note():
    shared.test_effects_character_brief_is_top_note(gen="v3")
