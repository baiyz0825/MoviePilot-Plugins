from tests.v2.subtitlestudio import test_pipeline as shared


def test_generation_uses_local_sidecar_and_skips_asr(tmp_path):
    shared.test_generation_uses_local_sidecar_and_skips_asr(tmp_path, gen="v3")


def test_strm_never_calls_asr(tmp_path):
    shared.test_strm_never_calls_asr(tmp_path, gen="v3")


def test_effects_character_brief_is_top_note():
    shared.test_effects_character_brief_is_top_note(gen="v3")
