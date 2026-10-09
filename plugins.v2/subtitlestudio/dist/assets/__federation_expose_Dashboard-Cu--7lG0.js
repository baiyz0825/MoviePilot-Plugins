import { importShared } from './__federation_fn_import-JrT3xvdd.js';
import { c as createStudioApi } from './studioApi-CSoBhxaG.js';

const {createTextVNode:_createTextVNode,resolveComponent:_resolveComponent,withCtx:_withCtx,createVNode:_createVNode,toDisplayString:_toDisplayString,openBlock:_openBlock,createBlock:_createBlock} = await importShared('vue');


const {onMounted,ref} = await importShared('vue');


const _sfc_main = {
  __name: 'Dashboard',
  props: {
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  config: { type: Object, default: () => ({}) },
  allowRefresh: { type: Boolean, default: true },
},
  setup(__props, { expose: __expose }) {

const props = __props;

const counts = ref({});
const pluginApi = createStudioApi(props.api, `plugin/${props.pluginId}`);

async function load() {
  try {
    const data = await pluginApi.status();
    counts.value = data?.counts || {};
  } catch {
    counts.value = {};
  }
}

onMounted(load);
__expose({ load });

return (_ctx, _cache) => {
  const _component_VCardTitle = _resolveComponent("VCardTitle");
  const _component_VCardText = _resolveComponent("VCardText");
  const _component_VCard = _resolveComponent("VCard");

  return (_openBlock(), _createBlock(_component_VCard, { class: "ss-card" }, {
    default: _withCtx(() => [
      _createVNode(_component_VCardTitle, null, {
        default: _withCtx(() => [...(_cache[0] || (_cache[0] = [
          _createTextVNode("字幕工坊队列", -1)
        ]))]),
        _: 1
      }),
      _createVNode(_component_VCardText, null, {
        default: _withCtx(() => [
          _createTextVNode(" 运行中 " + _toDisplayString(counts.value.running || 0) + " · 等待 " + _toDisplayString(counts.value.pending || 0) + " · 失败 " + _toDisplayString(counts.value.failed || 0), 1)
        ]),
        _: 1
      })
    ]),
    _: 1
  }))
}
}

};

export { _sfc_main as default };
