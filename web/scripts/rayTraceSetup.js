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
let lightingUniformBuffer;
let clearRayOutputPipeline;
let clearRayOutputBindGroup;

async function loadShader(url) {
    const response = await fetch(url);
    return await response.text();
}

async function setupRayTracePipeline(device, outVoxelsBuffer, brickMetadataBuffer, staticBricksBuffer, sceneUniformBuffer, cameraUniformBuffer, voxelBufferSize, brickRequestBuffer, paletteBuffer, lightingOptions = {}) {
    // Load shaders
    const rayTraceCode = await loadShader("shaders/rayTrace.wgsl");
    const rayTraceModule = device.createShaderModule({ code: rayTraceCode });

    const clearRayOutputCode = await loadShader("shaders/clearRayOutput.wgsl");
    const clearRayOutputModule = device.createShaderModule({ code: clearRayOutputCode });
    // Logging removed per request

    // Create output buffer for ray tracing results (RGBA8 image)
    const canvasWidth = document.getElementById("gpuCanvas").width;
    const canvasHeight = document.getElementById("gpuCanvas").height;
    const rayTraceOutputSize = canvasWidth * canvasHeight * 4; // u32 per pixel = 4 bytes

    rayTraceOutputBuffer = device.createBuffer({
        size: rayTraceOutputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

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
    const ambient = lightingOptions.ambient ?? 0.5;
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
            { binding: 0, resource: { buffer: outVoxelsBuffer } },
            { binding: 1, resource: { buffer: rayTraceOutputBuffer } },
            { binding: 2, resource: { buffer: brickMetadataBuffer } },  // No size/offset
            { binding: 3, resource: { buffer: staticBricksBuffer } },   // No size/offset
            { binding: 4, resource: { buffer: sceneUniformBuffer } },
            { binding: 5, resource: { buffer: cameraUniformBuffer } },
            { binding: 6, resource: { buffer: screenUniformBuffer } },
            { binding: 7, resource: { buffer: brickRequestBuffer } },   // No size/offset
            { binding: 8, resource: { buffer: paletteBuffer } },        // Palette for LOD 0 bricks
            { binding: 9, resource: { buffer: lightingUniformBuffer } }
        ]
    });

    // Create clear ray output pipeline
    clearRayOutputPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: clearRayOutputModule,
            entryPoint: "clearRayOutput"
        }
    });

    // Create clear ray output bind group
    clearRayOutputBindGroup = device.createBindGroup({
        layout: clearRayOutputPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: rayTraceOutputBuffer } },
            { binding: 1, resource: { buffer: screenUniformBuffer } }
        ]
    });

    // Logging removed per request

    return {
        rayTracePipeline,
        rayTraceBindGroup,
        rayTraceOutputBuffer,
        rayTraceOutputSize: { width: canvasWidth, height: canvasHeight },
        screenUniformBuffer,
        lightingUniformBuffer,
        clearRayOutputPipeline,
        clearRayOutputBindGroup
    };
}

export { setupRayTracePipeline };
