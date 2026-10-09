from tests.v2.subtitlestudio import test_media_catalog as shared


def test_history_expands_fileitem_and_files_json():
    shared.test_history_expands_fileitem_and_files_json(gen="v3")


def test_catalog_groups_history_for_manual_submit(tmp_path):
    shared.test_catalog_groups_history_for_manual_submit(tmp_path, gen="v3")
