import { useCallback, useEffect, useRef, useState } from 'react'
import { Button } from '../components/ui/button'
import { getDailySummarySettings, runDailySummary, streamAgentMessage, updateDailySummarySettings } from '../lib/api'

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

type InlinePart = {
  type: 'text' | 'strong' | 'em' | 'code'
  text: string
}

const formatInline = (text: string): InlinePart[] => {
  const parts: InlinePart[] = []
  const pattern = /(\*\*([^*]+)\*\*|__([^_]+)__|`([^`]+)`|(?<!\*)\*([^*\n]+)\*(?!\*)|_([^_\n]+)_)/g
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push({ type: 'text', text: text.slice(lastIndex, match.index) })
    if (match[2] || match[3]) parts.push({ type: 'strong', text: match[2] || match[3] })
    else if (match[4]) parts.push({ type: 'code', text: match[4] })
    else parts.push({ type: 'em', text: match[5] || match[6] })
    lastIndex = pattern.lastIndex
  }

  if (lastIndex < text.length) parts.push({ type: 'text', text: text.slice(lastIndex) })
  return parts.length ? parts : [{ type: 'text', text }]
}

const renderFormattedMessage = (content: string) => {
  const lines = content.split('\n')
  return lines.map((line, lineIndex) => {
    const unordered = line.match(/^\s*[-*]\s+(.+)$/)
    const ordered = line.match(/^\s*(\d+)\.\s+(.+)$/)
    const quote = line.match(/^\s*>\s?(.+)$/)
    const display = unordered?.[1] ?? ordered?.[2] ?? quote?.[1] ?? line
    const className = quote ? 'border-l-2 border-slate-300 pl-3 text-slate-600' : unordered || ordered ? 'pl-4' : undefined
    const prefix = unordered ? '• ' : ordered ? `${ordered[1]}. ` : ''

    return (
      <p key={lineIndex} className={className || (line ? undefined : 'h-4')}>
        {prefix}
        {formatInline(display).map((part, partIndex) => {
          if (part.type === 'strong') return <strong key={partIndex}>{part.text}</strong>
          if (part.type === 'em') return <em key={partIndex}>{part.text}</em>
          if (part.type === 'code') return <code key={partIndex} className="rounded bg-slate-100 px-1 py-0.5 text-[0.95em] text-slate-800">{part.text}</code>
          return <span key={partIndex}>{part.text}</span>
        })}
      </p>
    )
  })
}

export default function Home() {
  const [userId, setUserId] = useState('demo-user')
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Hi — I can answer with memory-aware context while keeping the interface focused on chat.' }
  ])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [summaryEnabled, setSummaryEnabled] = useState(false)
  const [summaryTime, setSummaryTime] = useState('09:00')
  const [summaryTimezone, setSummaryTimezone] = useState('Asia/Singapore')
  const [telegramChatId, setTelegramChatId] = useState('')
  const [settingsStatus, setSettingsStatus] = useState('')
  const activeStream = useRef<AbortController | null>(null)

  useEffect(() => {
    const savedUserId = window.localStorage.getItem('memos-q-user-id')
    if (savedUserId) setUserId(savedUserId)
  }, [])

  useEffect(() => {
    window.localStorage.setItem('memos-q-user-id', userId)
  }, [userId])

  useEffect(() => {
    getDailySummarySettings(userId)
      .then((settings) => {
        setSummaryEnabled(settings.enabled)
        setSummaryTime(settings.summary_time)
        setSummaryTimezone(settings.timezone)
        setTelegramChatId(settings.telegram_chat_id || '')
      })
      .catch(() => undefined)
  }, [userId])

  const saveSummarySettings = useCallback(async () => {
    setSettingsStatus('Saving…')
    try {
      await updateDailySummarySettings(userId, { enabled: summaryEnabled, summary_time: summaryTime, timezone: summaryTimezone, telegram_chat_id: telegramChatId || null })
      setSettingsStatus('Saved')
    } catch (err) {
      setSettingsStatus(String(err))
    }
  }, [summaryEnabled, summaryTime, summaryTimezone, telegramChatId, userId])

  const triggerDailySummary = useCallback(async () => {
    setSettingsStatus('Generating summary…')
    try {
      const result = await runDailySummary(userId)
      setSettingsStatus(result.sent_to_telegram ? 'Summary sent to Telegram' : (result.error_message || 'Summary generated but not sent'))
    } catch (err) {
      setSettingsStatus(String(err))
    }
  }, [userId])

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
    <main className="flex min-h-screen flex-col bg-[#f7f7f4] text-slate-950">
      <header className="sticky top-0 z-10 flex h-16 items-center justify-center border-b border-slate-200 bg-white/85 px-5 backdrop-blur">
        <div className="absolute left-1/2 -translate-x-1/2 text-xl font-bold tracking-tight">MemOS-Q</div>
        <input aria-label="User ID" className="ml-auto w-36 rounded-full border border-slate-200 bg-white px-3 py-2 text-sm" value={userId} onChange={(event) => setUserId(event.target.value)} />
      </header>
      <section className="flex min-h-[calc(100vh-4rem)] flex-1 flex-col">
        <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-5 px-4 py-8 pb-36">
          {messages.map((item, index) => (
            <div key={index} className={item.role === 'user' ? 'ml-auto max-w-[80%] rounded-3xl bg-slate-900 px-5 py-3 leading-relaxed text-white' : 'message-content mr-auto max-w-[80%] rounded-3xl bg-white px-5 py-3 leading-relaxed shadow-sm'}>
              {item.content ? renderFormattedMessage(item.content) : (loading ? <span className="animate-pulse text-slate-400">Thinking…</span> : null)}
            </div>
          ))}
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="font-semibold">Daily Telegram Summary</h2>
                <p className="text-sm text-slate-600">Get a proactive 9 AM reflection with topics, memories, and follow-ups.</p>
              </div>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={summaryEnabled} onChange={(event) => setSummaryEnabled(event.target.checked)} />
                Enable
              </label>
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <input aria-label="Summary time" className="rounded-xl border border-slate-200 px-3 py-2 text-sm" value={summaryTime} onChange={(event) => setSummaryTime(event.target.value)} placeholder="09:00" />
              <input aria-label="Summary timezone" className="rounded-xl border border-slate-200 px-3 py-2 text-sm" value={summaryTimezone} onChange={(event) => setSummaryTimezone(event.target.value)} placeholder="Asia/Singapore" />
              <input aria-label="Telegram chat ID" className="rounded-xl border border-slate-200 px-3 py-2 text-sm" value={telegramChatId} onChange={(event) => setTelegramChatId(event.target.value)} placeholder="Telegram chat ID" />
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-3">
              <Button onClick={saveSummarySettings}>Save summary settings</Button>
              <Button onClick={triggerDailySummary}>Run now</Button>
              {settingsStatus && <span className="text-sm text-slate-600">{settingsStatus}</span>}
            </div>
          </div>
          {error && <p className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-900">{error}</p>}
        </div>
        <div className="fixed bottom-0 left-0 right-0 border-t border-slate-200 bg-[#f7f7f4]/95 p-4 backdrop-blur">
          <div className="mx-auto flex max-w-3xl items-end gap-3 rounded-3xl border border-slate-200 bg-white p-3 shadow-lg">
            <textarea className="max-h-40 min-h-12 flex-1 resize-none border-0 bg-transparent p-2 outline-none" value={message} onChange={(event) => setMessage(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submit() } }} placeholder="Message MemOS-Q" />
            <Button disabled={loading || !message.trim()} onClick={submit}>{loading ? 'Typing…' : 'Send'}</Button>
          </div>
        </div>
      </section>
    </main>
  )
}
