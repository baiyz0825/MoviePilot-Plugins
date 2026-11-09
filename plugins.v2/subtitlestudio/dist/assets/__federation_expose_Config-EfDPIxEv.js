import { importShared } from './__federation_fn_import-JrT3xvdd.js';
import { u as useHostInjects, a as useMobileViewport, _ as _sfc_main$1 } from './SettingsForm-CH7pq-Sc.js';
import { c as createStudioApi } from './studioApi-pMXn-9A1.js';

const {unref:_unref,createVNode:_createVNode,createTextVNode:_createTextVNode,resolveComponent:_resolveComponent,withCtx:_withCtx,createElementVNode:_createElementVNode,renderList:_renderList,Fragment:_Fragment,openBlock:_openBlock,createElementBlock:_createElementBlock,createBlock:_createBlock} = await importShared('vue');


const _hoisted_1 = { class: "plugin-root pa-3" };
const _hoisted_2 = { class: "ss-savebar d-flex ga-2 pa-3" };

const {onMounted,reactive,ref} = await importShared('vue');


const _sfc_main = {
  __name: 'Config',
  props: {
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  sourcePluginId: { type: String, default: 'SubtitleStudio' },
  initialConfig: { type: Object, default: () => ({}) },
},
  emits: ['save', 'close', 'switch'],
  setup(__props, { emit: __emit }) {

const props = __props;

const emit = __emit;
const { confirm, toast } = useHostInjects();
const isMobile = useMobileViewport();
const pluginApi = createStudioApi(props.api, `plugin/${props.pluginId || props.sourcePluginId || 'SubtitleStudio'}`);
const config = ref({ ...props.initialConfig });
const fields = ref([]);
const dirty = ref(false);
const modelPicker = reactive({ open: false, endpointId: '', items: [] });

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

async function onListModels(endpoint) {
  try {
    const data = await pluginApi.listModels(endpoint);
    const items = data?.items || [];
    if (!items.length) {
      toast.error?.('没有拉到模型');
      return
    }
    modelPicker.open = true;
    modelPicker.endpointId = endpoint.endpoint_id;
    modelPicker.items = items;
  } catch (error) {
    toast.error?.(error?.message || '拉模型失败');
  }
}

function applyModel(model) {
  config.value = {
    ...config.value,
    openai_endpoints: (config.value.openai_endpoints || []).map(item => (
      item.endpoint_id === modelPicker.endpointId ? { ...item, model } : item
    )),
  };
  dirty.value = true;
  modelPicker.open = false;
  toast.success?.(`已选模型 ${model}`);
}

return (_ctx, _cache) => {
  const _component_VBtn = _resolveComponent("VBtn");
  const _component_VCardTitle = _resolveComponent("VCardTitle");
  const _component_VListItem = _resolveComponent("VListItem");
  const _component_VList = _resolveComponent("VList");
  const _component_VCard = _resolveComponent("VCard");
  const _component_VDialog = _resolveComponent("VDialog");

  return (_openBlock(), _createElementBlock("div", _hoisted_1, [
    _createVNode(_sfc_main$1, {
      modelValue: config.value,
      "onUpdate:modelValue": _cache[0] || (_cache[0] = $event => ((config).value = $event)),
      fields: fields.value,
      mobile: _unref(isMobile),
      onDirty: _cache[1] || (_cache[1] = $event => (dirty.value = true)),
      onTestEndpoint: _cache[2] || (_cache[2] = $event => (_unref(pluginApi).testEndpoint($event))),
      onListModels: onListModels
    }, null, 8, ["modelValue", "fields", "mobile"]),
    _createElementVNode("div", _hoisted_2, [
      _createVNode(_component_VBtn, {
        color: "primary",
        class: "ss-touch",
        onClick: save
      }, {
        default: _withCtx(() => [...(_cache[5] || (_cache[5] = [
          _createTextVNode("保存", -1)
        ]))]),
        _: 1
      }),
      _createVNode(_component_VBtn, {
        variant: "text",
        class: "ss-touch",
        onClick: close
      }, {
        default: _withCtx(() => [...(_cache[6] || (_cache[6] = [
          _createTextVNode("关闭", -1)
        ]))]),
        _: 1
      }),
      _createVNode(_component_VBtn, {
        variant: "text",
        class: "ss-touch",
        onClick: _cache[3] || (_cache[3] = $event => (emit('switch')))
      }, {
        default: _withCtx(() => [...(_cache[7] || (_cache[7] = [
          _createTextVNode("打开全页", -1)
        ]))]),
        _: 1
      })
    ]),
    _createVNode(_component_VDialog, {
      modelValue: modelPicker.open,
      "onUpdate:modelValue": _cache[4] || (_cache[4] = $event => ((modelPicker.open) = $event)),
      "max-width": "420"
    }, {
      default: _withCtx(() => [
        _createVNode(_component_VCard, null, {
          default: _withCtx(() => [
            _createVNode(_component_VCardTitle, null, {
              default: _withCtx(() => [...(_cache[8] || (_cache[8] = [
                _createTextVNode("选择模型", -1)
              ]))]),
              _: 1
            }),
            _createVNode(_component_VList, null, {
              default: _withCtx(() => [
                (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(modelPicker.items, (item) => {
                  return (_openBlock(), _createBlock(_component_VListItem, {
                    key: item,
                    title: item,
                    onClick: $event => (applyModel(item))
                  }, null, 8, ["title", "onClick"]))
                }), 128))
              ]),
              _: 1
            })
          ]),
          _: 1
        })
      ]),
      _: 1
    }, 8, ["modelValue"])
  ]))
}
}

};

export { _sfc_main as default };
