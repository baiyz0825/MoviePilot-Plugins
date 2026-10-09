function unwrap(payload) {
  if (payload && typeof payload === 'object' && 'success' in payload) {
    if (!payload.success) {
      const error = new Error(payload.message || '请求失败');
      error.payload = payload;
      throw error
    }
    return payload.data
  }
  return payload
}

function createStudioApi(api, pluginBase) {
  const base = () => pluginBase.value || pluginBase || 'plugin/SubtitleStudio';
  const get = (path, params, extra = {}) => api.get(`${base()}${path}`, { params, ...extra }).then(unwrap);
  const post = (path, body, extra = {}) => api.post(`${base()}${path}`, body, extra).then(unwrap);
  const put = (path, body, extra = {}) => api.put(`${base()}${path}`, body, extra).then(unwrap);
  return {
    unwrap,
    status: () => get('/status', {}, { feedback: 'silent' }),
    config: () => get('/config', {}, { feedback: 'silent' }),
    saveConfig: body => post('/config', body, { feedback: 'all' }),
    fields: () => get('/fields', {}, { feedback: 'silent' }),
    media: (q = '', mediaType = '') => get('/media', { q, media_type: mediaType }, { feedback: 'silent' }),
    refreshMedia: () => post('/media/refresh', {}, { feedback: 'all' }),
    jobs: (q = '', status = '') => get('/jobs', { q, status }, { feedback: 'silent' }),
    createJob: body => post('/jobs', body),
    createJobs: body => post('/jobs/batch', body),
    job: id => get(`/jobs/${id}`, {}, { feedback: 'silent' }),
    cutIn: id => post(`/jobs/${id}/cut-in`, {}),
    setPriority: (id, priority) => post(`/jobs/${id}/priority`, { priority }),
    cancel: id => post(`/jobs/${id}/cancel`, {}),
    retry: id => post(`/jobs/${id}/retry`, {}),
    cues: id => get(`/jobs/${id}/cues`, {}, { feedback: 'silent' }),
    saveCue: (id, cueId, body) => put(`/jobs/${id}/cues/${cueId}`, body),
    exportJob: id => post(`/jobs/${id}/export`, {}),
    searchJob: id => post(`/jobs/${id}/search`, {}),
    testEndpoint: body => post('/endpoints/test', body, { feedback: 'all' }),
    listModels: body => post('/endpoints/models', body, { feedback: 'all' }),
    previewAssUrl: id => `${base()}/jobs/${id}/preview/ass`,
    previewVideoUrl: id => `${base()}/jobs/${id}/preview/video`,
  }
}

export { createStudioApi as c };
