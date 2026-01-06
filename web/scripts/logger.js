// Toggle logging via localStorage or URL query params.
// Examples:
//   localStorage.setItem('LOG', '*')
//   localStorage.setItem('LOG', 'WebGPU,RenderLoop')
//   localStorage.setItem('LOG_LEVEL', 'warn')
//   ?log=WebGPU,RenderLoop&log_level=warn

const levelRank = { info: 1, warn: 2, off: 3 };

function safeLocalStorageGet(key) {
    try { return window.localStorage?.getItem(key) ?? null; } catch (_) { return null; }
}

function safeLocalStorageSet(key, val) {
    try { window.localStorage?.setItem(key, val); } catch (_) { /* ignore */ }
}

function safeLocalStorageRemove(key) {
    try { window.localStorage?.removeItem(key); } catch (_) { /* ignore */ }
}

function parseList(val) {
    return (val ?? '')
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
}

const urlParams = (() => {
    try { return new URLSearchParams(window.location.search); }
    catch (_) { return new URLSearchParams(); }
})();

const initialFilter = parseList(urlParams.get('log') ?? safeLocalStorageGet('LOG'));
const initialLevel = (urlParams.get('log_level') ?? safeLocalStorageGet('LOG_LEVEL') ?? 'info').toLowerCase();

let runtimeConfig = {
    namespaces: initialFilter,
    level: initialLevel
};

function isNamespaceEnabled(ns) {
    if (runtimeConfig.namespaces.length === 0) return false;
    if (runtimeConfig.namespaces.includes('*')) return true;
    return runtimeConfig.namespaces.includes(ns);
}

function levelAllows(method) {
    const current = levelRank[runtimeConfig.level] ?? levelRank.info;
    return levelRank[method] >= current && current < levelRank.off;
}

class Logger {
    constructor(namespace) {
        this.namespace = namespace;
    }

    log(message, ...args) {
        if (!isNamespaceEnabled(this.namespace) || !levelAllows('info')) return;
        console.log(`[${this.namespace}] ${message}`, ...args);
    }

    warn(message, ...args) {
        if (!isNamespaceEnabled(this.namespace) || !levelAllows('warn')) return;
        console.warn(`[${this.namespace}] ${message}`, ...args);
    }
}

export function createLogger(namespace) {
    return new Logger(namespace);
}

// Programmatic toggle without reloading:
// setLogConfig({ namespaces: ['*'] | ['WebGPU', 'RenderLoop'], level: 'info'|'warn'|'off', persist: false })
export function setLogConfig({ namespaces, level, persist = false } = {}) {
    if (Array.isArray(namespaces)) {
        runtimeConfig.namespaces = namespaces;
        if (persist) {
            safeLocalStorageSet('LOG', namespaces.join(','));
        }
    }

    if (typeof level === 'string') {
        runtimeConfig.level = level.toLowerCase();
        if (persist) {
            safeLocalStorageSet('LOG_LEVEL', runtimeConfig.level);
        }
    }

    if (!persist) {
        // Clear persisted keys when user opts for runtime-only settings
        safeLocalStorageRemove('LOG');
        safeLocalStorageRemove('LOG_LEVEL');
    }
}
