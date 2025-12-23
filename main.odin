package main

import "core:fmt"
import "core:mem"
import "core:time"
import "core:math"
import "core:math/linalg"
import "vendor:sdl2"
import "vendor:wgpu"
import "vendor:wgpu/sdl2glue"

Vertex :: struct #packed {
    position: [4]f32,
    color:    [4]f32,
}

UniformBuffer :: struct #packed {
    mvp: matrix[4, 4]f32,
}

VERTICES := [3]Vertex{
    {position = {1.0, -1.0, 0.0, 1.0}, color = {1.0, 0.0, 0.0, 1.0}},
    {position = {-1.0, -1.0, 0.0, 1.0}, color = {0.0, 1.0, 0.0, 1.0}},
    {position = {0.0, 1.0, 0.0, 1.0}, color = {0.0, 0.0, 1.0, 1.0}},
}

INDICES := [3]u32{0, 1, 2}

SHADER_SOURCE :: `
struct Uniform {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> ubo: Uniform;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vertex_main(vert: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = vert.color;
    out.position = ubo.mvp * vert.position;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color);
}
`

DEPTH_FORMAT :: wgpu.TextureFormat.Depth32Float

State :: struct {
    instance:          wgpu.Instance,
    surface:           wgpu.Surface,
    adapter:           wgpu.Adapter,
    device:            wgpu.Device,
    queue:             wgpu.Queue,
    surface_config:    wgpu.SurfaceConfiguration,
    depth_texture:     wgpu.Texture,
    depth_view:        wgpu.TextureView,
    pipeline:          wgpu.RenderPipeline,
    vertex_buffer:     wgpu.Buffer,
    index_buffer:      wgpu.Buffer,
    uniform_buffer:    wgpu.Buffer,
    bind_group:        wgpu.BindGroup,
    bind_group_layout: wgpu.BindGroupLayout,
    model:             matrix[4, 4]f32,
    width:             u32,
    height:            u32,
    initialized:       bool,
}

main :: proc() {
    if sdl2.Init({.VIDEO}) != 0 {
        fmt.eprintln("Failed to initialize SDL2:", sdl2.GetError())
        return
    }
    defer sdl2.Quit()

    window := sdl2.CreateWindow(
        "Odin/WGPU Triangle",
        sdl2.WINDOWPOS_CENTERED,
        sdl2.WINDOWPOS_CENTERED,
        800,
        600,
        {.RESIZABLE},
    )
    if window == nil {
        fmt.eprintln("Failed to create window:", sdl2.GetError())
        return
    }
    defer sdl2.DestroyWindow(window)

    state: State
    state.width = 800
    state.height = 600
    init_wgpu(&state, window)
    defer cleanup(&state)

    last_time := time.now()
    running := true

    for running {
        event: sdl2.Event
        for sdl2.PollEvent(&event) {
            #partial switch event.type {
            case .QUIT:
                running = false
            case .KEYDOWN:
                if event.key.keysym.scancode == .ESCAPE {
                    running = false
                }
            case .WINDOWEVENT:
                if event.window.event == .RESIZED {
                    new_width := cast(u32)event.window.data1
                    new_height := cast(u32)event.window.data2
                    if new_width > 0 && new_height > 0 {
                        resize(&state, new_width, new_height)
                    }
                }
            }
        }

        if !state.initialized {
            continue
        }

        now := time.now()
        delta_time := f32(time.duration_seconds(time.diff(last_time, now)))
        last_time = now

        update(&state, delta_time)
        render(&state)
    }
}

init_wgpu :: proc(state: ^State, window: ^sdl2.Window) {
    state.instance = wgpu.CreateInstance(nil)
    if state.instance == nil {
        fmt.eprintln("Failed to create wgpu instance")
        return
    }

    state.surface = sdl2glue.GetSurface(state.instance, window)
    if state.surface == nil {
        fmt.eprintln("Failed to create surface")
        return
    }

    wgpu.InstanceRequestAdapter(
        state.instance,
        &{compatibleSurface = state.surface, powerPreference = .HighPerformance},
        {
            mode = .WaitAnyOnly,
            callback = proc "c" (
                status: wgpu.RequestAdapterStatus,
                adapter: wgpu.Adapter,
                message: wgpu.StringView,
                userdata1: rawptr,
                userdata2: rawptr,
            ) {
                context = {}
                s := cast(^State)userdata1

                if status != .Success {
                    fmt.eprintln("Failed to get adapter:", message)
                    return
                }

                s.adapter = adapter
                on_adapter(s)
            },
            userdata1 = state,
        },
    )
}

on_adapter :: proc(state: ^State) {
    wgpu.AdapterRequestDevice(
        state.adapter,
        nil,
        {
            mode = .WaitAnyOnly,
            callback = proc "c" (
                status: wgpu.RequestDeviceStatus,
                device: wgpu.Device,
                message: wgpu.StringView,
                userdata1: rawptr,
                userdata2: rawptr,
            ) {
                context = {}
                s := cast(^State)userdata1

                if status != .Success {
                    fmt.eprintln("Failed to get device:", message)
                    return
                }

                s.device = device
                on_device(s)
            },
            userdata1 = state,
        },
    )
}

on_device :: proc(state: ^State) {
    state.queue = wgpu.DeviceGetQueue(state.device)

    caps, caps_status := wgpu.SurfaceGetCapabilities(state.surface, state.adapter)
    if caps_status != .Success {
        fmt.eprintln("Failed to get surface capabilities")
        return
    }
    surface_format := caps.formats[0]

    state.surface_config = {
        device      = state.device,
        usage       = {.RenderAttachment},
        format      = surface_format,
        width       = state.width,
        height      = state.height,
        presentMode = .Fifo,
        alphaMode   = caps.alphaModes[0],
    }
    wgpu.SurfaceConfigure(state.surface, &state.surface_config)

    create_depth_texture(state)
    create_buffers(state)
    create_pipeline(state, surface_format)

    state.model = 1
    state.initialized = true
}

create_depth_texture :: proc(state: ^State) {
    if state.depth_texture != nil {
        wgpu.TextureDestroy(state.depth_texture)
    }
    if state.depth_view != nil {
        wgpu.TextureViewRelease(state.depth_view)
    }

    state.depth_texture = wgpu.DeviceCreateTexture(
        state.device,
        &{
            size = {state.width, state.height, 1},
            mipLevelCount = 1,
            sampleCount = 1,
            dimension = ._2D,
            format = DEPTH_FORMAT,
            usage = {.RenderAttachment},
        },
    )

    state.depth_view = wgpu.TextureCreateView(state.depth_texture, nil)
}

create_buffers :: proc(state: ^State) {
    state.vertex_buffer = wgpu.DeviceCreateBufferWithData(
        state.device,
        &{label = "Vertex Buffer", usage = {.Vertex}},
        VERTICES[:],
    )

    state.index_buffer = wgpu.DeviceCreateBufferWithData(
        state.device,
        &{label = "Index Buffer", usage = {.Index}},
        INDICES[:],
    )

    state.uniform_buffer = wgpu.DeviceCreateBuffer(
        state.device,
        &{
            label = "Uniform Buffer",
            size = size_of(UniformBuffer),
            usage = {.Uniform, .CopyDst},
        },
    )

    state.bind_group_layout = wgpu.DeviceCreateBindGroupLayout(
        state.device,
        &{
            entryCount = 1,
            entries = &wgpu.BindGroupLayoutEntry{
                binding = 0,
                visibility = {.Vertex},
                buffer = {type = .Uniform},
            },
        },
    )

    state.bind_group = wgpu.DeviceCreateBindGroup(
        state.device,
        &{
            layout = state.bind_group_layout,
            entryCount = 1,
            entries = &wgpu.BindGroupEntry{
                binding = 0,
                buffer = state.uniform_buffer,
                size = size_of(UniformBuffer),
            },
        },
    )
}

create_pipeline :: proc(state: ^State, surface_format: wgpu.TextureFormat) {
    shader_module := wgpu.DeviceCreateShaderModule(
        state.device,
        &{
            nextInChain = &wgpu.ShaderSourceWGSL{
                sType = .ShaderSourceWGSL,
                code = SHADER_SOURCE,
            },
        },
    )
    defer wgpu.ShaderModuleRelease(shader_module)

    pipeline_layout := wgpu.DeviceCreatePipelineLayout(
        state.device,
        &{
            bindGroupLayoutCount = 1,
            bindGroupLayouts = &state.bind_group_layout,
        },
    )
    defer wgpu.PipelineLayoutRelease(pipeline_layout)

    vertex_attributes := [2]wgpu.VertexAttribute{
        {format = .Float32x4, offset = 0, shaderLocation = 0},
        {format = .Float32x4, offset = size_of([4]f32), shaderLocation = 1},
    }

    blend_state := wgpu.BlendState{
        color = {srcFactor = .SrcAlpha, dstFactor = .OneMinusSrcAlpha, operation = .Add},
        alpha = {srcFactor = .One, dstFactor = .OneMinusSrcAlpha, operation = .Add},
    }

    color_target := wgpu.ColorTargetState{
        format = surface_format,
        blend = &blend_state,
        writeMask = wgpu.ColorWriteMaskFlags_All,
    }

    vertex_buffer_layout := wgpu.VertexBufferLayout{
        arrayStride = size_of(Vertex),
        stepMode = .Vertex,
        attributeCount = 2,
        attributes = &vertex_attributes[0],
    }

    depth_stencil := wgpu.DepthStencilState{
        format = DEPTH_FORMAT,
        depthWriteEnabled = .True,
        depthCompare = .Less,
    }

    fragment_state := wgpu.FragmentState{
        module = shader_module,
        entryPoint = "fragment_main",
        targetCount = 1,
        targets = &color_target,
    }

    state.pipeline = wgpu.DeviceCreateRenderPipeline(
        state.device,
        &{
            layout = pipeline_layout,
            vertex = {
                module = shader_module,
                entryPoint = "vertex_main",
                bufferCount = 1,
                buffers = &vertex_buffer_layout,
            },
            primitive = {
                topology = .TriangleList,
                frontFace = .CW,
                cullMode = .None,
            },
            depthStencil = &depth_stencil,
            multisample = {
                count = 1,
                mask = 0xFFFFFFFF,
            },
            fragment = &fragment_state,
        },
    )
}

resize :: proc(state: ^State, width: u32, height: u32) {
    if !state.initialized {
        return
    }
    state.width = width
    state.height = height
    state.surface_config.width = width
    state.surface_config.height = height
    wgpu.SurfaceConfigure(state.surface, &state.surface_config)
    create_depth_texture(state)
}

update :: proc(state: ^State, delta_time: f32) {
    aspect := f32(state.width) / f32(max(state.height, 1))

    projection := linalg.matrix4_perspective_f32(
        math.to_radians(f32(80.0)),
        aspect,
        0.1,
        1000.0,
    )

    view := linalg.matrix4_look_at_f32(
        {0.0, 0.0, 3.0},
        {0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
    )

    rotation := linalg.matrix4_rotate_f32(
        math.to_radians(f32(30.0)) * delta_time,
        {0.0, 1.0, 0.0},
    )
    state.model = rotation * state.model

    uniform := UniformBuffer{
        mvp = projection * view * state.model,
    }

    wgpu.QueueWriteBuffer(
        state.queue,
        state.uniform_buffer,
        0,
        &uniform,
        size_of(UniformBuffer),
    )
}

render :: proc(state: ^State) {
    surface_texture := wgpu.SurfaceGetCurrentTexture(state.surface)
    #partial switch surface_texture.status {
    case .SuccessOptimal, .SuccessSuboptimal:
    case:
        return
    }

    surface_view := wgpu.TextureCreateView(surface_texture.texture, nil)
    defer wgpu.TextureViewRelease(surface_view)

    encoder := wgpu.DeviceCreateCommandEncoder(state.device, nil)

    color_attachment := wgpu.RenderPassColorAttachment{
        view = surface_view,
        loadOp = .Clear,
        storeOp = .Store,
        clearValue = {0.19, 0.24, 0.42, 1.0},
        depthSlice = wgpu.DEPTH_SLICE_UNDEFINED,
    }

    depth_attachment := wgpu.RenderPassDepthStencilAttachment{
        view = state.depth_view,
        depthLoadOp = .Clear,
        depthStoreOp = .Store,
        depthClearValue = 1.0,
    }

    render_pass := wgpu.CommandEncoderBeginRenderPass(
        encoder,
        &{
            colorAttachmentCount = 1,
            colorAttachments = &color_attachment,
            depthStencilAttachment = &depth_attachment,
        },
    )

    wgpu.RenderPassEncoderSetPipeline(render_pass, state.pipeline)
    wgpu.RenderPassEncoderSetBindGroup(render_pass, 0, state.bind_group, nil)
    wgpu.RenderPassEncoderSetVertexBuffer(render_pass, 0, state.vertex_buffer, 0, size_of(VERTICES))
    wgpu.RenderPassEncoderSetIndexBuffer(render_pass, state.index_buffer, .Uint32, 0, size_of(INDICES))
    wgpu.RenderPassEncoderDrawIndexed(render_pass, len(INDICES), 1, 0, 0, 0)

    wgpu.RenderPassEncoderEnd(render_pass)
    wgpu.RenderPassEncoderRelease(render_pass)

    command_buffer := wgpu.CommandEncoderFinish(encoder, nil)
    wgpu.CommandEncoderRelease(encoder)

    wgpu.QueueSubmit(state.queue, {command_buffer})
    wgpu.CommandBufferRelease(command_buffer)

    wgpu.SurfacePresent(state.surface)
    wgpu.TextureRelease(surface_texture.texture)
}

cleanup :: proc(state: ^State) {
    if state.pipeline != nil do wgpu.RenderPipelineRelease(state.pipeline)
    if state.bind_group != nil do wgpu.BindGroupRelease(state.bind_group)
    if state.bind_group_layout != nil do wgpu.BindGroupLayoutRelease(state.bind_group_layout)
    if state.uniform_buffer != nil do wgpu.BufferRelease(state.uniform_buffer)
    if state.index_buffer != nil do wgpu.BufferRelease(state.index_buffer)
    if state.vertex_buffer != nil do wgpu.BufferRelease(state.vertex_buffer)
    if state.depth_view != nil do wgpu.TextureViewRelease(state.depth_view)
    if state.depth_texture != nil do wgpu.TextureRelease(state.depth_texture)
    if state.queue != nil do wgpu.QueueRelease(state.queue)
    if state.device != nil do wgpu.DeviceRelease(state.device)
    if state.adapter != nil do wgpu.AdapterRelease(state.adapter)
    if state.surface != nil do wgpu.SurfaceRelease(state.surface)
    if state.instance != nil do wgpu.InstanceRelease(state.instance)
}
