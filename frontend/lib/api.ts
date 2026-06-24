const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

export type IntegrationStatus = {
  frontend: { nextjs_url: string }
  backend: { fastapi: boolean; qwen_agent_available?: boolean; qwen_agent_package_available?: boolean }
  models: {
    qwen_api_key_configured: boolean
    reasoning_model?: string
    flash_model?: string
    vision_model?: string
    embedding_model?: string
    embedding_dimensions?: number
    live_embeddings_required?: boolean
    base_url?: string
  }
  storage: { postgres_dsn_configured: boolean; redis_url_configured: boolean; s3_bucket?: string; pinecone_configured?: boolean }
  jobs?: { celery_broker_url_configured: boolean; celery_result_backend_configured?: boolean; celery_configured?: boolean }
  monitoring?: { langfuse_configured: boolean; langfuse_host?: string; otel_endpoint: string; otel_configured?: boolean; prometheus_metrics_path: string }
}

export async function getIntegrationStatus(): Promise<IntegrationStatus> {
  const response = await fetch(`${API_BASE_URL}/integrations/status`)
  if (!response.ok) {
    throw new Error(`Failed to load integration status: ${response.status}`)
  }
  return response.json()
}

export async function sendAgentMessage(userId: string, message: string) {
  const response = await fetch(`${API_BASE_URL}/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-user-id': userId },
    body: JSON.stringify({ message })
  })
  if (!response.ok) {
    throw new Error(await response.text())
  }
  return response.json()
}

export type MemoryRecord = {
  id: string
  content: string
  memory_type: string
  status: string
  confidence_score: number
  created_at: string
  updated_at: string
  tags: string[]
}

export async function getMemories(userId: string): Promise<MemoryRecord[]> {
  const response = await fetch(`${API_BASE_URL}/users/me/memories?include_inactive=true`, { headers: { 'x-user-id': userId } })
  if (!response.ok) throw new Error(await response.text())
  return response.json()
}

export async function approveMemory(userId: string, memoryId: string) {
  const response = await fetch(`${API_BASE_URL}/users/me/memories/${memoryId}/approve`, { method: 'POST', headers: { 'x-user-id': userId } })
  if (!response.ok) throw new Error(await response.text())
  return response.json()
}

export async function rejectMemory(userId: string, memoryId: string) {
  const response = await fetch(`${API_BASE_URL}/users/me/memories/${memoryId}/reject`, { method: 'POST', headers: { 'x-user-id': userId } })
  if (!response.ok) throw new Error(await response.text())
  return response.json()
}

export async function editMemory(userId: string, memoryId: string, content: string) {
  const response = await fetch(`${API_BASE_URL}/users/me/memories/${memoryId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', 'x-user-id': userId },
    body: JSON.stringify({ content })
  })
  if (!response.ok) throw new Error(await response.text())
  return response.json()
}

export async function deleteMemory(userId: string, memoryId: string) {
  const response = await fetch(`${API_BASE_URL}/users/me/memories/${memoryId}`, { method: 'DELETE', headers: { 'x-user-id': userId } })
  if (!response.ok) throw new Error(await response.text())
  return response.json()
}
