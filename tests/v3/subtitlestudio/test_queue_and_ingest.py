from tests.v2.subtitlestudio import test_queue_and_ingest as shared


def test_debounce_blocks_same_identity_within_five_minutes():
    shared.test_debounce_blocks_same_identity_within_five_minutes(gen="v3")


def test_gates_skip_existing_chinese(tmp_path):
    shared.test_gates_skip_existing_chinese(tmp_path, gen="v3")


def test_job_store_cut_in_does_not_touch_running(tmp_path):
    shared.test_job_store_cut_in_does_not_touch_running(tmp_path, gen="v3")


def test_scheduler_notifies_skipped_and_cancelled(tmp_path):
    shared.test_scheduler_notifies_skipped_and_cancelled(tmp_path, gen="v3")


def test_offpeak_window_cross_midnight():
    shared.test_offpeak_window_cross_midnight(gen="v3")


def test_search_html_parser_ignores_search_links():
    shared.test_search_html_parser_ignores_search_links(gen="v3")


def test_translation_partial_accept_and_no_placeholder():
    shared.test_translation_partial_accept_and_no_placeholder(gen="v3")
