import type { components } from './generated/api';

export const API = '/api/v1';
export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };
export type JsonObject = { [key: string]: Json };
export interface ValidationIssue {
  row?: number;
  row_index?: number;
  column?: string;
  field?: string;
  code?: string;
  reason_code?: string;
  message: string;
}
export class ApiError extends Error {
  constructor(
    message: string,
    public issues: ValidationIssue[] = [],
    public status?: number,
  ) {
    super(message);
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(API + path, init);
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    let message = `Request failed (${response.status}).`;
    let issues: ValidationIssue[] = [];
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = body.detail;
      if (typeof detail === 'string') message = detail;
      else if (detail && typeof detail === 'object') {
        if ('message' in detail && typeof detail.message === 'string') message = detail.message;
        if ('issues' in detail && Array.isArray(detail.issues))
          issues = detail.issues as ValidationIssue[];
      }
    }
    throw new ApiError(message, issues, response.status);
  }
  return response.json() as Promise<T>;
}
export function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}
export function numberAt(data: JsonObject, key: string): number | null {
  return typeof data[key] === 'number' ? data[key] : null;
}
export function objectAt(data: JsonObject, key: string): JsonObject {
  const value = data[key];
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
}
export function arrayAt(data: JsonObject, key: string): Json[] {
  return Array.isArray(data[key]) ? data[key] : [];
}
export function textAt(data: JsonObject, key: string, fallback = 'Not recorded'): string {
  return typeof data[key] === 'string' ? data[key] : fallback;
}
export interface ModelInfo {
  bundle_id: string;
  model_id: string;
  purpose: string;
  score_kind: string;
  limitation: string;
  policy: JsonObject;
  quality_summary: JsonObject;
}
export type Prediction = components['schemas']['PredictionResponse'];
export type Batch = components['schemas']['BatchResponse'];
export interface ExperimentData {
  fold_results: Json[];
  selection: JsonObject;
  final_metrics: JsonObject | null;
  final_status: string;
  limitation: string;
}
export interface StreamState extends JsonObject {
  stream_id: string;
  scenario: string;
  cursor: number;
  total_records: number;
}
