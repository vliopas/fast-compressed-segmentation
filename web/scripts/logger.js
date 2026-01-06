// Global logging control - set to false to disable all cache logs
const LOGGING_ENABLED = true;

class Logger {
    constructor(namespace) {
        this.namespace = namespace;
    }
    
    log(message, ...args) {
        if (!LOGGING_ENABLED) return;
        console.log(`[${this.namespace}] ${message}`, ...args);
    }
    
    warn(message, ...args) {
        if (!LOGGING_ENABLED) return;
        console.warn(`[${this.namespace}] ${message}`, ...args);
    }
}

export function createLogger(namespace) {
    return new Logger(namespace);
}
