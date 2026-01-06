import { loadDatasetFromArrayBuffer } from "./datasetLoader.js";
import { decodeAllRansBricks } from './decode.js';

let device;
let context;

let pipeline;
let bindGroup;
let computePipeline;
let computeBindGroup;

let vertexBuffer;
let uniformBuffer;
let storageTexture;

let canvas;
let dataset;

async function loadShader(url) {
    const response = await fetch(url);
    return await response.text();
}

async function initWebGPU() {
    // Check for WebGPU support
    if (!navigator.gpu) {
        console.error("WebGPU not supported!");
        return;
    }

    // Get GPU adapter and device
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        console.error("No GPU adapter found. Check your browser and hardware.");
        return;
    }

    device = await adapter.requestDevice();
    console.log("WebGPU device ready!", device);

    // Get canvas context
    canvas = document.getElementById("gpuCanvas");
    context = canvas.getContext("webgpu");

    // Configure swap chain
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: format,
        alphaMode: 'opaque'
    });

    // Vertex data for a triangle (x, y positions)
    const vertices = new Float32Array([
        0.0,  0.5,   // top
       -0.5, -0.5,   // bottom left
        0.5, -0.5    // bottom right
    ]);

    // Vertex buffer for quad (two triangles)
    const quadVertices = new Float32Array([
    -1,-1,  1,-1,  -1,1,  // first triangle
        1,-1,  1,1,   -1,1   // second triangle
    ]);


    // Create vertex buffer
    vertexBuffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(vertexBuffer, 0, vertices);

    // Uniform buffer
    uniformBuffer = device.createBuffer({
        size: 16, // 2 floats (angle, scale) = 8 bytes, align to 16
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Storage texture for compute pass output
    storageTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    });

    // Load shader code
    const vertexCode = await loadShader("shaders/vertex.wgsl");
    const fragmentCode = await loadShader("shaders/fragment.wgsl");

    // Simple WGSL shaders
    const shaderModule = device.createShaderModule({
        code: vertexCode + "\n" + fragmentCode
    });

    // Pipeline setup
    pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: shaderModule,
            entryPoint: "vs_main",
            buffers: [{
                arrayStride: 2 * 4,
                attributes: [{
                    shaderLocation: 0,
                    offset: 0,
                    format: "float32x2"
                }]
            }]
        },
        fragment: {
            module: shaderModule,
            entryPoint: "fs_main",
            targets: [{ format: format }]
        },
        primitive: {
            topology: "triangle-list"
        }
    });

    // Bind group
    bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: storageTexture.createView() }
        ]
    });

    // Compute pipeline
    const computeCode = await loadShader("shaders/compute.wgsl");
    const computeModule = device.createShaderModule({ code: computeCode });

    computePipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: computeModule, entryPoint: "cs_main" }
    });

    // Bind group for storage texture
    computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: storageTexture.createView() }]
    });

    requestAnimationFrame(frame);
}

function frame() {
    // Update uniforms
    const data = new Float32Array([angle, scale]);
    device.queue.writeBuffer(uniformBuffer, 0, data);

    const encoder = device.createCommandEncoder();

    // ---- Compute Pass ----
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(
        Math.ceil(canvas.width / 8),
        Math.ceil(canvas.height / 8)
    );
    computePass.end();

    // ---- Render Pass ----
    const view = context.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
            view,
            clearValue: { r: 1, g: 0, b: 0, a: 1 },
            loadOp: "load",
            storeOp: "store"
        }]
    });

    // Set pipeline and triangle
    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.draw(3);
    renderPass.end();

    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(frame);
}


async function loadFixedDataset() {
    const response = await fetch("../encoder/CSVEncoder/compressed_dataset.csbd");

    if (!response.ok) {
        throw new Error("Failed to load dataset.bin");
    }

    const arrayBuffer = await response.arrayBuffer();
    dataset = loadDatasetFromArrayBuffer(arrayBuffer);

    // console.log("Dataset loaded successfully:", dataset);

    // Store globally or pass to WebGPU pipeline
    window.dataset = dataset;

    return dataset;
}

async function initApp() {
    try {
        // Load dataset
        const dataset = await loadFixedDataset(); // this returns dataset
        
        // Decode bricks (rANS) once
        const decodedBricks = await decodeAllRansBricks(dataset);
        decodedBricks.forEach((decoded, i) => {
            dataset.bricks[i].encodedData = decoded;      // update each brick
            dataset.bricks[i].encodedSize = decoded.length; // update size
        });
        
        console.log("Dataset loaded:", dataset);

        // Initialize WebGPU with decoded bricks
        initWebGPU(decodedBricks);

    } catch (err) {
        console.error("Error loading or decoding dataset:", err);
    }
}

let angle = 0;
let scale = 1;

window.addEventListener("keydown", (e) => {
    if (e.key === "q" || e.key === "Q") angle += 0.05;
    if (e.key === "e" || e.key === "E") angle -= 0.05;
});

window.addEventListener("wheel", (e) => {
    scale += e.deltaY * -0.001;
    scale = Math.max(0.1, scale); // clamp
});

// Call the async init function
initApp();