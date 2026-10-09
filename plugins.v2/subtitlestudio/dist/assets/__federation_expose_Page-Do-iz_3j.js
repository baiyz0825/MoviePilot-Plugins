import { importShared } from './__federation_fn_import-JrT3xvdd.js';
import _sfc_main$1 from './__federation_expose_AppPage-Bh9OsMh3.js';

const {createElementVNode:_createElementVNode,resolveComponent:_resolveComponent,createVNode:_createVNode,withCtx:_withCtx,mergeProps:_mergeProps,openBlock:_openBlock,createElementBlock:_createElementBlock} = await importShared('vue');


const {ref} = await importShared('vue');


const _sfc_main = {
  __name: 'Page',
  props: {
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  sourcePluginId: { type: String, default: 'SubtitleStudio' },
  navKey: { type: String, default: 'main' },
},
  emits: ['close', 'action', 'switch'],
  setup(__props, { emit: __emit }) {



const emit = __emit;
const pageRef = ref(null);

return (_ctx, _cache) => {
  const _component_VSpacer = _resolveComponent("VSpacer");
  const _component_VBtn = _resolveComponent("VBtn");
  const _component_VToolbar = _resolveComponent("VToolbar");
  const _component_VDivider = _resolveComponent("VDivider");

  return (_openBlock(), _createElementBlock("div", null, [
    _createVNode(_component_VToolbar, { density: "comfortable" }, {
      default: _withCtx(() => [
        _cache[3] || (_cache[3] = _createElementVNode("div", { class: "text-h6 ms-3" }, "字幕工坊", -1)),
        _createVNode(_component_VSpacer),
        _createVNode(_component_VBtn, {
          icon: "mdi-refresh",
          variant: "text",
          onClick: _cache[0] || (_cache[0] = $event => (pageRef.value?.reload()))
        }),
        _createVNode(_component_VBtn, {
          icon: "mdi-close",
          variant: "text",
          onClick: _cache[1] || (_cache[1] = $event => (emit('close')))
        })
      ]),
      _: 1
    }),
    _createVNode(_component_VDivider),
    _createVNode(_sfc_main$1, _mergeProps({
      ref_key: "pageRef",
      ref: pageRef
    }, _ctx.$props, {
      "hide-title": "",
      onAction: _cache[2] || (_cache[2] = $event => (emit('action')))
    }), null, 16)
  ]))
}
}

};

export { _sfc_main as default };
