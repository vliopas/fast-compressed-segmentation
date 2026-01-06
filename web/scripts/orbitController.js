// Orbit controller and dataset bounds utilities.
// - Drag (pointer): rotates yaw (left/right) and pitch (up/down) around the target.
// - Wheel: zooms in/out, clamped between minDistance and maxDistance so the camera stays outside the dataset.
// - Camera is kept looking at the target center at all times.

function normalize(v) {
    const len = Math.hypot(v.x, v.y, v.z) || 1;
    return { x: v.x / len, y: v.y / len, z: v.z / len };
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

// Compute dataset bounds to center the orbit camera
export function computeDatasetBounds(dataset) {
    const min = { x: Infinity, y: Infinity, z: Infinity };
    const max = { x: -Infinity, y: -Infinity, z: -Infinity };
    const brickSize = dataset.header?.brickSize ?? 1;

    for (const brick of dataset.bricks) {
        min.x = Math.min(min.x, brick.position.x);
        min.y = Math.min(min.y, brick.position.y);
        min.z = Math.min(min.z, brick.position.z);
        max.x = Math.max(max.x, brick.position.x + brickSize);
        max.y = Math.max(max.y, brick.position.y + brickSize);
        max.z = Math.max(max.z, brick.position.z + brickSize);
    }

    const size = {
        x: max.x - min.x,
        y: max.y - min.y,
        z: max.z - min.z
    };

    const center = {
        x: min.x + size.x * 0.5,
        y: min.y + size.y * 0.5,
        z: min.z + size.z * 0.5
    };

    return { min, max, size, center };
}

// Orbit controller keeps camera focused on a target with clamped pitch and radius
export class OrbitController {
    constructor({
        canvas,
        camera,
        target,
        minDistance,
        maxDistance,
        maxPitch = Math.PI / 3,
        rotateSpeed = 0.003,
        zoomSpeed = 0.15,
        initialYaw = 0,
        initialPitch = 0,
        initialRadius = null
    }) {
        this.canvas = canvas;
        this.camera = camera;
        this.target = target;
        this.minDistance = minDistance;
        this.maxDistance = maxDistance;
        this.maxPitch = maxPitch;
        this.rotateSpeed = rotateSpeed;
        this.zoomSpeed = zoomSpeed;

        this.yaw = initialYaw;
        this.pitch = clamp(initialPitch, -maxPitch, maxPitch);
        this.radius = clamp(initialRadius ?? maxDistance * 0.6, minDistance, maxDistance);
        this.dragging = false;
        this.capturedPointer = null;
        this.pendingUpload = true;

        // Bind handlers to the instance
        this.onPointerDown = (e) => this.handlePointerDown(e);
        this.onPointerMove = (e) => this.handlePointerMove(e);
        this.onPointerUp = (e) => this.handlePointerUp(e);
        this.onWheel = (e) => this.handleWheel(e);

        this.attach();
        this.apply();
    }

    attach() {
        this.canvas.addEventListener('pointerdown', this.onPointerDown);
        window.addEventListener('pointermove', this.onPointerMove);
        window.addEventListener('pointerup', this.onPointerUp);
        this.canvas.addEventListener('wheel', this.onWheel, { passive: false });
    }

    detach() {
        this.canvas.removeEventListener('pointerdown', this.onPointerDown);
        window.removeEventListener('pointermove', this.onPointerMove);
        window.removeEventListener('pointerup', this.onPointerUp);
        this.canvas.removeEventListener('wheel', this.onWheel);
    }

    apply() {
        this.pitch = clamp(this.pitch, -this.maxPitch, this.maxPitch);
        this.radius = clamp(this.radius, this.minDistance, this.maxDistance);

        const cp = Math.cos(this.pitch);
        const sp = Math.sin(this.pitch);
        const cy = Math.cos(this.yaw);
        const sy = Math.sin(this.yaw);

        const position = {
            x: this.target.x + this.radius * cp * sy,
            y: this.target.y + this.radius * sp,
            z: this.target.z + this.radius * cp * cy
        };

        const direction = normalize({
            x: this.target.x - position.x,
            y: this.target.y - position.y,
            z: this.target.z - position.z
        });

        this.camera.update(position, direction);
        this.pendingUpload = true;
    }

    handlePointerDown(e) {
        this.dragging = true;
        this.capturedPointer = e.pointerId;
        this.canvas.setPointerCapture(e.pointerId);
    }

    handlePointerMove(e) {
        if (!this.dragging) return;
        this.yaw -= e.movementX * this.rotateSpeed;
        this.pitch -= e.movementY * this.rotateSpeed;
        this.apply();
    }

    handlePointerUp(e) {
        if (this.capturedPointer !== null) {
            this.canvas.releasePointerCapture(this.capturedPointer);
            this.capturedPointer = null;
        }
        this.dragging = false;
    }

    handleWheel(e) {
        // Only zoom if shift + scroll
        if (!e.shiftKey) return;
        
        e.preventDefault();
        const delta = Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 1000);
        const scale = Math.exp(delta * this.zoomSpeed * 0.0015);
        this.radius *= scale;
        this.apply();
    }

    needsUpload() {
        return this.pendingUpload || this.camera.needsUpload();
    }

    markUploaded() {
        this.pendingUpload = false;
        this.camera.markUploaded();
    }

    dispose() {
        this.detach();
    }

    getState() {
        return { yaw: this.yaw, pitch: this.pitch, radius: this.radius };
    }
}

//
