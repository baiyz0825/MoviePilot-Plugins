from tests.v2.subtitlestudio import test_domain as shared


def test_multiline_paths_use_real_newlines_only():
    shared.test_multiline_paths_use_real_newlines_only(gen="v3")


def test_identity_pair_and_v3_required():
    from tests.subtitlestudio_support.loader import load_domain
    identity = load_domain("v3", "core.identity")
    pair = identity.identity_from_v3_pair("themoviedb", "693134")
    assert pair["media_source"] == "themoviedb"
    assert pair["media_id"] == "693134"
    empty = identity.identity_from_v3_pair("themoviedb", "0")
    assert empty["media_source"] == ""
    assert identity.build_media_key(empty["media_source"], empty["media_id"]) == ""


def test_config_defaults_and_language_limit():
    shared.test_config_defaults_and_language_limit(gen="v3")


def test_notify_payload_and_gating():
    shared.test_notify_payload_and_gating(gen="v3")


def test_cuegraph_roundtrip_and_ass_styles():
    shared.test_cuegraph_roundtrip_and_ass_styles(gen="v3")


def test_format_repair_keeps_good_sentences():
    shared.test_format_repair_keeps_good_sentences(gen="v3")


def test_asr_repair_drops_empty_and_fixes_overlap():
    shared.test_asr_repair_drops_empty_and_fixes_overlap(gen="v3")


def test_export_plan_cartesian_and_extras():
    shared.test_export_plan_cartesian_and_extras(gen="v3")


def test_export_plan_uses_player_language_tags():
    shared.test_export_plan_uses_player_language_tags(gen="v3")


def test_packager_writes_and_skip_policy(tmp_path):
    shared.test_packager_writes_and_skip_policy(tmp_path, gen="v3")
