import { useCallback, useEffect, useMemo, useState } from 'react'
import { Button } from '../components/ui/button'
import { Card } from '../components/ui/card'
import { approveMemory, deleteMemory, editMemory, getMemories, MemoryRecord, rejectMemory } from '../lib/api'

const tabs = [
  ['Active', 'active'],
  ['Pending Review', 'pending_review'],
  ['Conflicts', 'possibly_conflicting'],
  ['Rejected', 'rejected'],
  ['Forgotten', 'forgotten']
]

export default function MemoriesPage() {
  const [userId, setUserId] = useState('demo-user')
  const [memories, setMemories] = useState<MemoryRecord[]>([])
  const [tab, setTab] = useState('active')
  const [query, setQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    try {
      setMemories(await getMemories(userId))
      setError('')
    } catch (err) {
      setError(String(err))
    }
  }, [userId])

  useEffect(() => {
    load()
  }, [load])

  const filtered = useMemo(() => {
    return memories.filter((memory) => {
      const status = statusFilter || tab
      return (
        memory.status === status &&
        (!typeFilter || memory.memory_type === typeFilter) &&
        (!query || memory.content.toLowerCase().includes(query.toLowerCase()) || memory.tags.join(' ').includes(query.toLowerCase()))
      )
    })
  }, [memories, query, statusFilter, tab, typeFilter])

  const act = async (fn: () => Promise<unknown>) => {
    try {
      await fn()
      await load()
    } catch (err) {
      setError(String(err))
    }
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-6 py-8">
      <section>
        <p className="mb-3 text-sm font-semibold uppercase tracking-[0.3em] text-accent">Memory Management</p>
        <h1 className="text-4xl font-black">My Memories</h1>
      </section>
      <Card>
        <div className="grid gap-3 md:grid-cols-4">
          <input className="rounded-xl border border-white/10 bg-black/30 p-3" value={userId} onChange={(e) => setUserId(e.target.value)} />
          <input className="rounded-xl border border-white/10 bg-black/30 p-3" placeholder="Keyword search" value={query} onChange={(e) => setQuery(e.target.value)} />
          <input className="rounded-xl border border-white/10 bg-black/30 p-3" placeholder="Type filter" value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)} />
          <input className="rounded-xl border border-white/10 bg-black/30 p-3" placeholder="Status filter" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} />
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          {tabs.map(([label, value]) => (
            <Button key={value} onClick={() => setTab(value)} className={tab === value ? '' : 'bg-white/10'}>{label}</Button>
          ))}
        </div>
      </Card>
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[980px] text-left text-sm">
            <thead className="text-slate-300">
              <tr><th>Content</th><th>Type</th><th>Status</th><th>Confidence</th><th>Created At</th><th>Updated At</th><th>Tags</th><th>Actions</th></tr>
            </thead>
            <tbody>
              {filtered.map((memory) => (
                <tr key={memory.id} className="border-t border-white/10 align-top">
                  <td className="max-w-md py-3">{memory.content}</td>
                  <td>{memory.memory_type}</td>
                  <td>{memory.status}</td>
                  <td>{memory.confidence_score.toFixed(2)}</td>
                  <td>{new Date(memory.created_at).toLocaleString()}</td>
                  <td>{new Date(memory.updated_at).toLocaleString()}</td>
                  <td>{memory.tags.join(', ')}</td>
                  <td className="flex flex-wrap gap-2 py-3">
                    <Button onClick={() => act(() => approveMemory(userId, memory.id))}>Approve</Button>
                    <Button onClick={() => act(() => rejectMemory(userId, memory.id))}>Reject</Button>
                    <Button onClick={() => { const content = window.prompt('Edit memory', memory.content); if (content) act(() => editMemory(userId, memory.id, content)) }}>Edit</Button>
                    <Button onClick={() => act(() => deleteMemory(userId, memory.id))}>Delete</Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {error && <p className="mt-4 rounded-xl bg-red-400/10 p-4 text-red-100">{error}</p>}
      </Card>
    </main>
  )
}
