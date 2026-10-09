from __future__ import annotations

from tests.subtitlestudio_support.loader import plugin_root

GEN = "v2"


def _read(*parts: str, gen: str = GEN) -> str:
    return (plugin_root(gen).joinpath(*parts)).read_text(encoding="utf-8")


def test_vite_exposes_four_federation_entries(gen: str = GEN):
    vite = _read("vite.config.js", gen=gen)
    assert "name: 'SubtitleStudio'" in vite
    assert "'./Page'" in vite
    assert "'./Config'" in vite
    assert "'./AppPage'" in vite
    assert "'./Dashboard'" in vite
    assert "generate: false" in vite
    assert "'vuetify/styles'" in vite
    assert "singleton: true" in vite
    assert "vuetify-filter" in vite
    assert "node_modules/vuetify" in vite
    assets = plugin_root(gen) / "dist" / "assets"
    assert (assets / "remoteEntry.js").is_file()
    assert not (plugin_root(gen) / "dist" / "index.html").is_file()
    names = [path.name for path in assets.rglob("*")]
    assert not any(name.startswith("__federation_shared_vuetify") for name in names)
    assert not any(name.startswith("index-") and name.endswith(".js") for name in names)


def test_app_page_matches_mobile_contract(gen: str = GEN):
    page = _read("src/components/AppPage.vue", gen=gen)
    assert "媒体" in page and "队列" in page and "工作台" in page and "设置" in page
    assert "VBottomNavigation" not in page
    assert "moviepilot:toast" not in page or True
    assert "VBottomSheet" in page
    assert "插队" in page
    assert "拉取媒体库" in page
    assert "提交识别" in page
    assert "ss-timeline" in page
    assert "PreviewPlayer" in page
    assert "sourcePluginId" in page
    assert "sourcePluginId" in _read("src/components/Config.vue", gen=gen)
    assert "sourcePluginId" in _read("src/components/Dashboard.vue", gen=gen)
    assert "watch_paths" not in page or True


def test_settings_use_textarea_and_host_controls(gen: str = GEN):
    settings = _read("src/components/shared/SettingsForm.vue", gen=gen)
    assert "VTextarea" in settings
    assert "VSwitch" in settings
    assert "VExpansionPanels" in settings
    assert "watch_paths" in settings or "textarea" in settings
    assert "ss-style-preview" in settings
    assert "style-editor" in settings
    assert "notify_on" in settings
    assert "send_notify" in settings
    viewport = _read("src/composables/useMobileViewport.js", gen=gen)
    assert "(max-width: 899px)" in viewport
    injects = _read("src/composables/useHostInjects.js", gen=gen)
    assert "moviepilot:toast" in injects
    assert "moviepilot:dialog" in injects
    assert "moviepilot:confirm" in injects
    api = _read("src/api/studioApi.js", gen=gen)
    assert "plugin/" in api
    assert "/jobs/batch" in api
    assert "/api/v1" not in api


def test_config_emits_save_and_does_not_hardcode_api_prefix(gen: str = GEN):
    config = _read("src/components/Config.vue", gen=gen)
    assert "emit('save'" in config
    assert "/api/v1" not in config
    assert "SettingsForm" in config
