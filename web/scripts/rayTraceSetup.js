// Ray tracing specific setup
// 
// Shader specialization support:
// The rayTrace shader supports specialization constants for compile-time optimizations:
// - GRADIENT_SHADING_ENABLED: Enable/disable gradient-based normal shading
// - AO_ENABLED: Enable/disable ambient occlusion with shadow rays
// 
// To use specialization constants (GPU feature level 3+):
//   const shaderModule = device.createShaderModule({ code: rayTraceCode });
//   const pipeline = device.createComputePipeline({
//       layout: "auto",
//       compute: {
//           module: shaderModule,
//           entryPoint: "rayTrace",
//           constants: {
//               "GRADIENT_SHADING_ENABLED": true,    // or false for flat shading
//               "AO_ENABLED": true                   // or false to skip AO
//           }
//       }
//   });

let rayTracePipeline;
let rayTraceBindGroup;
let rayTraceOutputBuffer;
let screenUniformBuffer;
let aoHistoryBuffer;
let lightingUniformBuffer;

async function loadShader(url) {
    const response = await fetch(url);
    return await response.text();
}

async function setupRayTracePipeline(device, outVoxelsBuffer, brickMetadataBuffer, staticBricksBuffer, sceneUniformBuffer, cameraUniformBuffer, voxelBufferSize, brickRequestBuffer, lightingOptions = {}) {
    // Load ray tracing shader
    const rayTraceCode = await loadShader("shaders/rayTrace.wgsl");
    const rayTraceModule = device.createShaderModule({ code: rayTraceCode });

    // Create output buffer for ray tracing results (RGBA8 image)
    const canvasWidth = document.getElementById("gpuCanvas").width;
    const canvasHeight = document.getElementById("gpuCanvas").height;
    const rayTraceOutputSize = canvasWidth * canvasHeight * 4; // u32 per pixel = 4 bytes

    rayTraceOutputBuffer = device.createBuffer({
        size: rayTraceOutputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    // AO history buffer (f16 per pixel for 50% bandwidth savings!)
    // initialized to fully visible (1.0)
    const aoHistorySize = canvasWidth * canvasHeight * 2; // f16 = 2 bytes
    aoHistoryBuffer = device.createBuffer({
        size: aoHistorySize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    // Initialize with f16 values (using Uint16Array with f16 bit pattern for 1.0)
    // f16 bit pattern for 1.0 is 0x3C00
    const aoInit = new Uint16Array(canvasWidth * canvasHeight).fill(0x3C00);
    device.queue.writeBuffer(aoHistoryBuffer, 0, aoInit);

    // Create screen uniform (width, height, aspect, padding)
    const aspect = canvasHeight === 0 ? 1 : canvasWidth / canvasHeight;
    const screenData = new Float32Array([canvasWidth, canvasHeight, aspect, 0]);
    screenUniformBuffer = device.createBuffer({
        size: screenData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(screenUniformBuffer, 0, screenData);

    // Lighting + AO parameters
    const lightAngle = lightingOptions.lightAngle ?? 25;
    const angleRad = (lightAngle * Math.PI) / 180;
    const lightDir = [
        Math.sin(angleRad),
        0.6,
        -Math.cos(angleRad)
    ];
    const ambient = lightingOptions.ambient ?? 0.2;
    const shadowAlphaThreshold = lightingOptions.shadowAlphaThreshold ?? 0.2;
    const aoBlend = lightingOptions.aoBlend ?? 0.1;
    const aoStrength = lightingOptions.aoStrength ?? 0.5;
    const gradientShadingEnabled = lightingOptions.gradientShadingEnabled ?? true;
    const diffuseStrength = lightingOptions.diffuseStrength ?? 0.8;

    const lightingData = new Float32Array([
        lightDir[0], lightDir[1], lightDir[2], 0.0,
        ambient, shadowAlphaThreshold, aoBlend, aoStrength,
        gradientShadingEnabled ? 1.0 : 0.0, diffuseStrength, 0.0, 0.0
    ]);
    lightingUniformBuffer = device.createBuffer({
        size: 48,  // 3 vec4s now
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(lightingUniformBuffer, 0, lightingData);

    // Create ray tracing pipeline
    rayTracePipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: rayTraceModule,
            entryPoint: "rayTrace"
        }
    });

    // Create ray tracing bind group
    rayTraceBindGroup = device.createBindGroup({
        layout: rayTracePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: outVoxelsBuffer } },      // Decoded voxel colors
            { binding: 1, resource: { buffer: rayTraceOutputBuffer } }, // Output image
            { binding: 2, resource: { buffer: brickMetadataBuffer } },  // Dynamic brick info (offset, LOD)
            { binding: 3, resource: { buffer: staticBricksBuffer } },   // Static brick info (isEmpty flag, palette, stream)
            { binding: 4, resource: { buffer: sceneUniformBuffer } },   // Scene constants (brickSize)
            { binding: 5, resource: { buffer: cameraUniformBuffer } },  // Camera (position, direction, fov)
            { binding: 6, resource: { buffer: screenUniformBuffer } },  // Screen info (width, height, aspect)
            { binding: 7, resource: { buffer: brickRequestBuffer } },   // GPU writes brick access requests
            { binding: 8, resource: { buffer: aoHistoryBuffer } },      // Temporal AO history
            { binding: 9, resource: { buffer: lightingUniformBuffer } } // Lighting + AO params
        ]
    });

    return {
        rayTracePipeline,
        rayTraceBindGroup,
        rayTraceOutputBuffer,
        rayTraceOutputSize: { width: canvasWidth, height: canvasHeight },
        screenUniformBuffer,
        aoHistoryBuffer,
        lightingUniformBuffer
    };
}

export { setupRayTracePipeline };
