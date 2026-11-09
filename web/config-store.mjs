import {readFile, rename, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {parse, stringify} from 'yaml';

const SECTION_FIELDS = {
  llm: ['model_provider', 'model', 'base_url'],
  image: ['model_provider', 'model', 'base_url'],
  video: ['model_provider', 'model', 'base_url'],
  audio: ['model_provider', 'model', 'base_url', 'voice', 'bgm_file', 'bgm_volume'],
  subtitles: ['enabled', 'language', 'burn_in', 'font', 'font_size'],
  embedding: ['model_provider', 'model', 'base_url'],
  reranker: ['model', 'base_url'],
};

export async function readAgentConfig(repoRoot) {
  const {payload} = await loadConfig(repoRoot);
  return publicConfig(payload);
}

export async function saveAgentConfig(repoRoot, input) {
  if (!input || typeof input !== 'object' || !input.sections || typeof input.sections !== 'object') {
    throw new Error('Configuration sections are required');
  }
  const {configPath, payload} = await loadConfig(repoRoot);
  for (const [section, fields] of Object.entries(SECTION_FIELDS)) {
    const update = input.sections[section];
    if (!update || typeof update !== 'object') continue;
    const current = payload[section] && typeof payload[section] === 'object' ? payload[section] : {};
    for (const field of fields) {
      if (!(field in update)) continue;
      if (section === 'subtitles' && (field === 'enabled' || field === 'burn_in')) {
        current[field] = validatedBoolean(update[field], `${section}.${field}`);
      } else if (section === 'subtitles' && field === 'font_size') {
        current[field] = validatedInteger(update[field], `${section}.${field}`, 12, 120);
      } else {
        current[field] = validatedValue(update[field], `${section}.${field}`, 2_048);
      }
    }
    if (section !== 'subtitles' && typeof update.api_key === 'string' && update.api_key.trim()) {
      current.api_key = validatedValue(update.api_key, `${section}.api_key`, 8_192);
    }
    // Handle character voices mapping for audio section
    if (section === 'audio') {
      if (typeof update.character_voices_text === 'string') {
        const charMap = {};
        for (const line of update.character_voices_text.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith('#')) continue;
          const idx = trimmed.indexOf('=') !== -1 ? trimmed.indexOf('=') : trimmed.indexOf(':');
          if (idx !== -1) {
            const charName = trimmed.slice(0, idx).trim();
            const voiceId = trimmed.slice(idx + 1).trim();
            if (charName && voiceId) charMap[charName] = voiceId;
          }
        }
        current.character_voices = charMap;
      } else if (update.character_voices && typeof update.character_voices === 'object') {
        current.character_voices = update.character_voices;
      }
    }
    payload[section] = current;
  }
  const temporaryPath = `${configPath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, stringify(payload, {lineWidth: 0}), {mode: 0o600});
  await rename(temporaryPath, configPath);
  return publicConfig(payload);
}

async function loadConfig(repoRoot) {
  const configPath = path.join(repoRoot, 'configs', 'agent.local.yaml');
  let text = '';
  try {
    text = await readFile(configPath, 'utf8');
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  const payload = text ? parse(text) : {};
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('configs/agent.local.yaml must be a YAML mapping');
  }
  return {configPath, payload};
}

function publicConfig(payload) {
  const sections = {};
  for (const [section, fields] of Object.entries(SECTION_FIELDS)) {
    const source = payload[section] && typeof payload[section] === 'object' ? payload[section] : {};
    if (section === 'subtitles') {
      sections[section] = {
        enabled: parseBool(source.enabled, false),
        language: typeof source.language === 'string' && source.language.trim() ? source.language.trim() : 'vi',
        burn_in: parseBool(source.burn_in, true),
        font: typeof source.font === 'string' && source.font.trim() ? source.font.trim() : 'Arial',
        font_size: source.font_size !== undefined && source.font_size !== null ? String(source.font_size) : '34',
      };
      continue;
    }
    const result = {};
    for (const field of fields) result[field] = typeof source[field] === 'string' ? source[field] : '';
    result.api_key = '';
    result.has_api_key = Boolean(typeof source.api_key === 'string' && source.api_key.trim());
    if (section === 'audio') {
      const cv = source.character_voices && typeof source.character_voices === 'object' ? source.character_voices : {};
      result.character_voices = cv;
      result.character_voices_text = Object.entries(cv)
        .map(([k, v]) => `${k} = ${v}`)
        .join('\n');
      result.voice = typeof source.voice === 'string' ? source.voice : '';
      result.bgm_file = typeof source.bgm_file === 'string' ? source.bgm_file : '';
      result.bgm_volume = source.bgm_volume !== undefined ? String(source.bgm_volume) : '0.12';
    }
    sections[section] = result;
  }
  return {sections};
}

function parseBool(value, fallback) {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return Boolean(value);
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
    if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
  }
  return fallback;
}

function validatedBoolean(value, label) {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
    if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
  }
  throw new Error(`${label} must be a boolean`);
}

function validatedInteger(value, label, min, max) {
  const num = typeof value === 'number' ? value : Number(String(value ?? '').trim());
  if (!Number.isInteger(num) || num < min || num > max) {
    throw new Error(`${label} must be an integer between ${min} and ${max}`);
  }
  return num;
}

function validatedValue(value, label, maxLength) {
  if (typeof value !== 'string') throw new Error(`${label} must be a string`);
  const normalized = value.trim();
  if (normalized.length > maxLength) throw new Error(`${label} is too long`);
  if (label.endsWith('.base_url') && normalized && !/^https?:\/\//i.test(normalized)) {
    throw new Error(`${label} must use http:// or https://`);
  }
  return normalized;
}
