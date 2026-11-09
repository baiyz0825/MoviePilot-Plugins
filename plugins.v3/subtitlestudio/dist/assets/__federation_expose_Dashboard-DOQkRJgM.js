import { importShared } from './__federation_fn_import-JrT3xvdd.js';
import { c as createStudioApi } from './studioApi-pMXn-9A1.js';

const {createElementVNode:_createElementVNode,createTextVNode:_createTextVNode,resolveComponent:_resolveComponent,withModifiers:_withModifiers,withCtx:_withCtx,openBlock:_openBlock,createBlock:_createBlock,createCommentVNode:_createCommentVNode,createVNode:_createVNode,toDisplayString:_toDisplayString} = await importShared('vue');


const {onBeforeUnmount,onMounted,ref} = await importShared('vue');


const _sfc_main = {
  __name: 'Dashboard',
  props: {
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  sourcePluginId: { type: String, default: 'SubtitleStudio' },
  config: { type: Object, default: () => ({}) },
  allowRefresh: { type: Boolean, default: true },
},
  emits: ['action'],
  setup(__props, { expose: __expose, emit: __emit }) {

const props = __props;

const emit = __emit;
const counts = ref({});
const pluginApi = createStudioApi(props.api, `plugin/${props.pluginId || props.sourcePluginId}`);
let timer = null;

async function load() {
  try {
    const data = await pluginApi.status();
    counts.value = data?.counts || {};
  } catch {
    counts.value = {};
  }
}

function openQueue() {
  emit('action', { page: 'AppPage', nav: 'jobs' });
}

onMounted(() => {
  load();
  if (props.allowRefresh) {
    timer = setInterval(load, 15000);
  }
});
onBeforeUnmount(() => {
  if (timer) clearInterval(timer);
});
__expose({ load });

return (_ctx, _cache) => {
  const _component_VBtn = _resolveComponent("VBtn");
  const _component_VCardTitle = _resolveComponent("VCardTitle");
  const _component_VCardText = _resolveComponent("VCardText");
  const _component_VCard = _resolveComponent("VCard");

  return (_openBlock(), _createBlock(_component_VCard, {
    class: "ss-card",
    role: "button",
    onClick: openQueue
  }, {
    default: _withCtx(() => [
      _createVNode(_component_VCardTitle, { class: "d-flex align-center justify-space-between" }, {
        default: _withCtx(() => [
          _cache[1] || (_cache[1] = _createElementVNode("span", null, "字幕工坊队列", -1)),
          (__props.allowRefresh)
            ? (_openBlock(), _createBlock(_component_VBtn, {
                key: 0,
                size: "small",
                variant: "text",
                class: "ss-touch",
                onClick: _withModifiers(load, ["stop"])
              }, {
                default: _withCtx(() => [...(_cache[0] || (_cache[0] = [
                  _createTextVNode("刷新", -1)
                ]))]),
                _: 1
              }))
            : _createCommentVNode("", true)
        ]),
        _: 1
      }),
      _createVNode(_component_VCardText, null, {
        default: _withCtx(() => [
          _createTextVNode(" 运行中 " + _toDisplayString(counts.value.running || 0) + " · 等待 " + _toDisplayString(counts.value.pending || 0) + " · 失败 " + _toDisplayString(counts.value.failed || 0) + " ", 1),
          _cache[2] || (_cache[2] = _createElementVNode("div", { class: "text-caption text-medium-emphasis mt-1" }, "点击打开工作台队列", -1))
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
