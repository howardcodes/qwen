const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

export type IntegrationStatus = {
  frontend: { nextjs_url: string }
  backend: { fastapi: boolean; qwen_agent_available?: boolean }
  models: {
    qwen_api_key_configured: boolean
    reasoning_model?: string
    flash_model?: string
    vision_model?: string
    base_url?: string
  }
  storage: { postgres_dsn_configured: boolean; redis_url_configured: boolean; s3_bucket?: string; pinecone_configured?: boolean }
  jobs?: { celery_broker_url_configured: boolean }
  monitoring?: { langfuse_configured: boolean; otel_endpoint: string; prometheus_metrics_path: string }
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
