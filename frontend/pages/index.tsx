import { useCallback, useRef, useState } from 'react'
import { Button } from '../components/ui/button'
import { streamAgentMessage } from '../lib/api'

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

const starterConversations = ['Memory chat', 'Study help', 'Preferences']

export default function Home() {
  const [userId, setUserId] = useState('demo-user')
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Hi — I can answer with memory-aware context while keeping the interface focused on chat.' }
  ])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const activeStream = useRef<AbortController | null>(null)

  const submit = useCallback(async () => {
    const trimmed = message.trim()
    if (!trimmed || loading) return
    activeStream.current?.abort()
    const controller = new AbortController()
    activeStream.current = controller
    setLoading(true)
    setError('')
    setMessage('')
    setMessages((current) => [...current, { role: 'user', content: trimmed }, { role: 'assistant', content: '' }])
    try {
      await streamAgentMessage(
        userId,
        trimmed,
        (token) => setMessages((current) => current.map((item, index) => index === current.length - 1 ? { ...item, content: item.content + token } : item)),
        { signal: controller.signal }
      )
    } catch (err) {
      if (!controller.signal.aborted) setError(String(err))
    } finally {
      if (activeStream.current === controller) {
        activeStream.current = null
        setLoading(false)
      }
    }
  }, [loading, message, userId])

  return (
    <main className="flex min-h-screen bg-[#f7f7f4] text-slate-950">
      <aside className="hidden w-72 shrink-0 border-r border-slate-200 bg-white/70 p-4 md:block">
        <div className="mb-6 text-lg font-bold">MemOS-Q</div>
        <div className="space-y-2">
          {starterConversations.map((title) => <div key={title} className="rounded-xl px-3 py-2 text-sm text-slate-700 hover:bg-slate-100">{title}</div>)}
        </div>
      </aside>
      <section className="flex min-h-screen flex-1 flex-col">
        <header className="flex h-16 items-center justify-between border-b border-slate-200 bg-white/80 px-5 backdrop-blur">
          <h1 className="text-base font-semibold">Memory-aware chat</h1>
          <input aria-label="User ID" className="w-36 rounded-full border border-slate-200 bg-white px-3 py-2 text-sm" value={userId} onChange={(event) => setUserId(event.target.value)} />
        </header>
        <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-5 px-4 py-8 pb-36">
          {messages.map((item, index) => (
            <div key={index} className={item.role === 'user' ? 'ml-auto max-w-[80%] rounded-3xl bg-slate-900 px-5 py-3 text-white' : 'mr-auto max-w-[80%] rounded-3xl bg-white px-5 py-3 shadow-sm'}>
              {item.content || (loading ? <span className="animate-pulse text-slate-400">Thinking…</span> : null)}
            </div>
          ))}
          {error && <p className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-900">{error}</p>}
        </div>
        <div className="fixed bottom-0 left-0 right-0 border-t border-slate-200 bg-[#f7f7f4]/95 p-4 backdrop-blur md:left-72">
          <div className="mx-auto flex max-w-3xl items-end gap-3 rounded-3xl border border-slate-200 bg-white p-3 shadow-lg">
            <textarea className="max-h-40 min-h-12 flex-1 resize-none border-0 bg-transparent p-2 outline-none" value={message} onChange={(event) => setMessage(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submit() } }} placeholder="Message MemOS-Q" />
            <Button disabled={loading || !message.trim()} onClick={submit}>{loading ? 'Typing…' : 'Send'}</Button>
          </div>
        </div>
      </section>
    </main>
  )
}
