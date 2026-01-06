// Setup display render pipeline to show raytraced output on canvas

export async function setupDisplayPipeline(device, rayTraceOutputBuffer, screenUniformBuffer, canvasFormat) {
    const displayShader = await fetch('shaders/display.wgsl').then(r => r.text());
    
    const displayModule = device.createShaderModule({
        label: 'Display Shader',
        code: displayShader
    });
    
    const displayPipeline = device.createRenderPipeline({
        label: 'Display Pipeline',
        layout: 'auto',
        vertex: {
            module: displayModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: displayModule,
            entryPoint: 'fragmentMain',
            targets: [{
                format: canvasFormat,
            }],
        },
        primitive: {
            topology: 'triangle-list',
        },
    });
    
    const displayBindGroup = device.createBindGroup({
        label: 'Display Bind Group',
        layout: displayPipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: { buffer: rayTraceOutputBuffer }
            },
            {
                binding: 1,
                resource: { buffer: screenUniformBuffer }
            }
        ],
    });
    
    return { displayPipeline, displayBindGroup };
}
