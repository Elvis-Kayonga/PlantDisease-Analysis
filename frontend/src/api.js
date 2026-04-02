const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

async function parseResponse(response) {
  const text = await response.text();
  let payload = {};

  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = {};
  }

  if (!response.ok) {
    const detail = payload.detail || `Request failed with status ${response.status}`;
    throw new Error(detail);
  }

  return payload;
}

export async function apiGet(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);
  return parseResponse(response);
}

export async function apiPost(path, formData) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    body: formData
  });
  return parseResponse(response);
}

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  return apiPost("/predict", formData);
}

export async function uploadData(files, className) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("class_name", className);
  return apiPost("/upload-data", formData);
}

export async function triggerRetrain() {
  const response = await fetch(`${API_BASE_URL}/retrain`, { method: "POST" });
  return parseResponse(response);
}

export { API_BASE_URL };
