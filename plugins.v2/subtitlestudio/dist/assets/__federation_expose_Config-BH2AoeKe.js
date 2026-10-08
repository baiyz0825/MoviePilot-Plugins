import { importShared } from './__federation_fn_import-JrT3xvdd.js';
import { u as useHostInjects, a as useMobileViewport, _ as _sfc_main$1 } from './SettingsForm-CGWuXqQW.js';
import { c as createStudioApi } from './studioApi-faEpD5op.js';

const {unref:_unref,createVNode:_createVNode,createTextVNode:_createTextVNode,resolveComponent:_resolveComponent,withCtx:_withCtx,createElementVNode:_createElementVNode,openBlock:_openBlock,createElementBlock:_createElementBlock} = await importShared('vue');


const _hoisted_1 = { class: "plugin-root pa-3" };
const _hoisted_2 = { class: "ss-savebar d-flex ga-2 pa-3" };

const {onMounted,ref} = await importShared('vue');


const _sfc_main = {
  __name: 'Config',
  props: {
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  initialConfig: { type: Object, default: () => ({}) },
},
  emits: ['save', 'close', 'switch'],
  setup(__props, { emit: __emit }) {

const props = __props;

const emit = __emit;
const { confirm } = useHostInjects();
const isMobile = useMobileViewport();
const pluginApi = createStudioApi(props.api, `plugin/${props.pluginId || 'SubtitleStudio'}`);
const config = ref({ ...props.initialConfig });
const fields = ref([]);
const dirty = ref(false);

onMounted(async () => {
  try {
    const data = await pluginApi.fields();
    fields.value = data.fields || [];
  } catch {
    fields.value = [];
  }
});

async function save() {
  emit('save', config.value);
}

async function close() {
  if (dirty.value && !(await confirm('有未保存的设置，确定离开？'))) return
  emit('close');
}

return (_ctx, _cache) => {
  const _component_VBtn = _resolveComponent("VBtn");

  return (_openBlock(), _createElementBlock("div", _hoisted_1, [
    _createVNode(_sfc_main$1, {
      modelValue: config.value,
      "onUpdate:modelValue": _cache[0] || (_cache[0] = $event => ((config).value = $event)),
      fields: fields.value,
      mobile: _unref(isMobile),
      onDirty: _cache[1] || (_cache[1] = $event => (dirty.value = true))
    }, null, 8, ["modelValue", "fields", "mobile"]),
    _createElementVNode("div", _hoisted_2, [
      _createVNode(_component_VBtn, {
        color: "primary",
        class: "ss-touch",
        onClick: save
      }, {
        default: _withCtx(() => [...(_cache[3] || (_cache[3] = [
          _createTextVNode("保存", -1)
        ]))]),
        _: 1
      }),
      _createVNode(_component_VBtn, {
        variant: "text",
        class: "ss-touch",
        onClick: close
      }, {
        default: _withCtx(() => [...(_cache[4] || (_cache[4] = [
          _createTextVNode("关闭", -1)
        ]))]),
        _: 1
      }),
      _createVNode(_component_VBtn, {
        variant: "text",
        class: "ss-touch",
        onClick: _cache[2] || (_cache[2] = $event => (emit('switch')))
      }, {
        default: _withCtx(() => [...(_cache[5] || (_cache[5] = [
          _createTextVNode("打开全页", -1)
        ]))]),
        _: 1
      })
    ])
  ]))
}
}

};

export { _sfc_main as default };
