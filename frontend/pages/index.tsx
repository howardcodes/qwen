import { useCallback, useEffect, useMemo, useState } from 'react'
import ReactFlow, { Background, Controls, Edge, Node } from 'reactflow'
import { Button } from '../components/ui/button'
import { Card } from '../components/ui/card'
import { getIntegrationStatus, IntegrationStatus, streamAgentMessage } from '../lib/api'

const nodes: Node[] = [
  { id: 'user', position: { x: 0, y: 90 }, data: { label: 'User Input' }, type: 'input' },
  { id: 'api', position: { x: 210, y: 90 }, data: { label: 'FastAPI + Qwen-Agent' } },
  { id: 'models', position: { x: 460, y: 0 }, data: { label: 'Qwen / DashScope' } },
  { id: 'memory', position: { x: 460, y: 180 }, data: { label: 'Postgres + Pinecone + Redis + MinIO' } },
  { id: 'jobs', position: { x: 760, y: 180 }, data: { label: 'Celery Maintenance' } },
  { id: 'monitoring', position: { x: 760, y: 0 }, data: { label: 'Langfuse + OTel + Prometheus + Grafana' }, type: 'output' }
]

const edges: Edge[] = [
  { id: 'e1', source: 'user', target: 'api', animated: true },
  { id: 'e2', source: 'api', target: 'models', animated: true },
  { id: 'e3', source: 'api', target: 'memory', animated: true },
  { id: 'e4', source: 'memory', target: 'jobs', animated: true },
  { id: 'e5', source: 'api', target: 'monitoring', animated: true }
]

export default function Home() {
  const [status, setStatus] = useState<IntegrationStatus | null>(null)
  const [userId, setUserId] = useState('demo-user')
  const [message, setMessage] = useState('What do you remember about me?')
  const [response, setResponse] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getIntegrationStatus().then(setStatus).catch((err) => setError(String(err)))
  }, [])

  const statusItems = useMemo(() => {
    if (!status) return []
    return [
      ['FastAPI', status.backend.fastapi],
      ['Qwen API key', status.models.qwen_api_key_configured],
      ['Qwen-Agent', Boolean(status.backend.qwen_agent_available)],
      ['Postgres on ECS', status.storage.postgres_dsn_configured],
      ['Pinecone', Boolean(status.storage.pinecone_configured)],
      ['Redis/Celery', Boolean(status.jobs?.celery_broker_url_configured)],
      ['Langfuse', Boolean(status.monitoring?.langfuse_configured)]
    ]
  }, [status])

  const submit = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setResponse('')
      await streamAgentMessage(userId, message, (token) => {
        setResponse((current) => current + token)
      })
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }, [message, userId])

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-8 px-6 py-8">
      <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div>
          <p className="mb-3 text-sm font-semibold uppercase tracking-[0.3em] text-accent">MemOS-Q Live Stack</p>
          <h1 className="text-5xl font-black leading-tight">Self-evolving memory OS powered by QwenCloud integrations.</h1>
          <p className="mt-5 max-w-2xl text-lg text-slate-300">
            This dashboard connects Next.js 12, Tailwind CSS, shadcn-style UI primitives, React Flow, FastAPI,
            Qwen/DashScope, Pinecone vector recall, ECS-hosted Postgres/Redis/MinIO, Celery maintenance, and observability endpoints.
          </p>
        </div>
        <Card>
          <h2 className="mb-4 text-xl font-bold">Integration status</h2>
          <div className="grid gap-3">
            {statusItems.map(([label, ok]) => (
              <div key={String(label)} className="flex items-center justify-between rounded-xl bg-white/5 px-4 py-3">
                <span>{label}</span>
                <span className={ok ? 'text-emerald-300' : 'text-amber-300'}>{ok ? 'configured' : 'needs key'}</span>
              </div>
            ))}
          </div>
        </Card>
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <Card className="h-[420px]">
          <h2 className="mb-3 text-xl font-bold">Memory architecture graph</h2>
          <ReactFlow nodes={nodes} edges={edges} fitView>
            <Background />
            <Controls />
          </ReactFlow>
        </Card>
        <Card>
          <h2 className="mb-3 text-xl font-bold">Live Qwen-Agent chat</h2>
          <label className="mb-2 block text-sm font-semibold text-slate-300">Authenticated user ID</label>
          <input
            className="mb-4 w-full rounded-xl border border-white/10 bg-black/30 p-3 outline-none focus:border-accent"
            value={userId}
            onChange={(event) => setUserId(event.target.value)}
          />
          <label className="mb-2 block text-sm font-semibold text-slate-300">Message</label>
          <textarea
            className="min-h-32 w-full rounded-xl border border-white/10 bg-black/30 p-4 outline-none focus:border-accent"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
          />
          <Button className="mt-4" disabled={loading} onClick={submit}>
            {loading ? 'Streaming from QwenCloud...' : 'Send to MemOS-Q'}
          </Button>
          {response && <p className="mt-4 rounded-xl bg-emerald-400/10 p-4 text-emerald-100">{response}</p>}
          {error && <p className="mt-4 rounded-xl bg-red-400/10 p-4 text-red-100">{error}</p>}
        </Card>
      </section>
    </main>
  )
}
