import { inject } from 'vue'

function fallbackToast() {
  return {
    success: message => console.info(message),
    error: message => console.error(message),
    info: message => console.info(message),
  }
}

export function useHostInjects() {
  const toast = inject('moviepilot:toast', fallbackToast())
  const dialog = inject('moviepilot:dialog', null)
  const confirm = inject('moviepilot:confirm', async () => window.confirm('确定？'))
  const nativeSubscribe = inject('moviepilot:nativeSubscribe', null)
  return { toast, dialog, confirm, nativeSubscribe }
}
