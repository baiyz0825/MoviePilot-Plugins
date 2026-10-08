from tests.v2.subtitlestudio import test_frontend as shared


def test_vite_exposes_four_federation_entries():
    shared.test_vite_exposes_four_federation_entries(gen="v3")


def test_app_page_matches_mobile_contract():
    shared.test_app_page_matches_mobile_contract(gen="v3")


def test_settings_use_textarea_and_host_controls():
    shared.test_settings_use_textarea_and_host_controls(gen="v3")


def test_config_emits_save_and_does_not_hardcode_api_prefix():
    shared.test_config_emits_save_and_does_not_hardcode_api_prefix(gen="v3")
