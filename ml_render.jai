#load "ml.jai";
#import "d3d_compiler";

InstanceVertex :: struct {
    offset : Vector3;
    colour : Vector3;
}

neuron_instance_data: [layer_0_size + layer_1_size + layer_2_size + layer_3_size] InstanceVertex;
wire_instance_data: [2 * (layer_0_weights.count + layer_1_weights.count + layer_2_weights.count)] InstanceVertex;

init_instance_positions :: ()
{    
    spacing :: 1.5;
    max_width :: (layer_0_dim - 1) * spacing;
    layer_1_padding :: 0.5 * (max_width - ((layer_1_dim - 1) * spacing));
    layer_2_padding :: 0.5 * (max_width - ((layer_2_dim - 1) * spacing));
    layer_3_padding_x :: 0.5 * (max_width - ((layer_3_size - 1) * spacing));
    layer_3_padding_y :: 0.5 * (max_width - spacing);
    for i : 0..(layer_0_dim-1) {
        for j : 0..(layer_0_dim-1) {
            neuron_instance_data[(i * layer_0_dim) + j].offset.x = xx (i * spacing);
            neuron_instance_data[(i * layer_0_dim) + j].offset.y = xx (j * spacing);
            neuron_instance_data[(i * layer_0_dim) + j].offset.z = 0;
        }
    }
    offset := layer_0_size;
    for i : 0..(layer_1_dim-1) {
        for j : 0..(layer_1_dim-1) {
            neuron_instance_data[offset + (i * layer_1_dim) + j].offset.x = xx layer_1_padding + (i * spacing);
            neuron_instance_data[offset + (i * layer_1_dim) + j].offset.y = xx layer_1_padding + (j * spacing);
            neuron_instance_data[offset + (i * layer_1_dim) + j].offset.z = 25;
        }
    }
    offset += layer_1_size;
    for i : 0..(layer_2_dim-1) {
        for j : 0..(layer_2_dim-1) {
            neuron_instance_data[offset + (i * layer_2_dim) + j].offset.x = xx layer_2_padding + (i * spacing);
            neuron_instance_data[offset + (i * layer_2_dim) + j].offset.y = xx layer_2_padding + (j * spacing);
            neuron_instance_data[offset + (i * layer_2_dim) + j].offset.z = 40;
        }
    }
    offset += layer_2_size;
    for i : 0..(layer_3_size-1) {
            neuron_instance_data[offset + i].offset.x = xx layer_3_padding_x + (i * spacing);
            neuron_instance_data[offset + i].offset.y = layer_3_padding_y;
            neuron_instance_data[offset + i].offset.z = 60;
    }    
    offset += layer_3_size;
    for i : 0..(layer_3_size-1) {

    }

    offset = 0;
    for i : 0..(layer_0_size-1) {
        for k : 0..(layer_1_size-1) {
            wire_instance_data[offset].offset = neuron_instance_data[i].offset;
            wire_instance_data[offset + 1].offset = neuron_instance_data[layer_0_size + k].offset;
            offset += 2;
        }
    }
    for i : 0..(layer_1_size-1) {
        for k : 0..(layer_2_size-1) {
            wire_instance_data[offset].offset = neuron_instance_data[layer_0_size + i].offset;
            wire_instance_data[offset + 1].offset = neuron_instance_data[layer_0_size + layer_1_size + k].offset;
            offset += 2;
        }
    }
    for i : 0..(layer_2_size-1) {
        for k : 0..(layer_3_size-1) {
            wire_instance_data[offset].offset = neuron_instance_data[layer_0_size + layer_1_size + i].offset;
            wire_instance_data[offset + 1].offset = neuron_instance_data[layer_0_size + layer_1_size + layer_2_size + k].offset;
            offset += 2;
        }
    }
}

init_ml :: ()
{
    init_weights();
    init_instance_positions();
}

neuron_shader_source :: #string END
struct VsInput
{
    float3 pos  : POSITION;
    float3 color   : COLOR;
};

struct VsInputInstanced
{
    float4 pos : POSITION;
    // per instance
    float3 offset  : POSITION1;
    float3 color   : COLOR1;
};

struct PsInput
{
    float4 pos : SV_POSITION;
    float3 pos_o : POSITION;
};

cbuffer cb : register(b0)
{
    matrix g_viewProj;
}

PsInput vs(VsInput input)
{
    PsInput output;
    output.pos_o = input.pos.xyz;
    output.pos = mul(g_viewProj, float4(input.pos.xyz, 1));    
    return output;
}

float4 ps(PsInput input) : SV_Target
{
    return float4(float3(1,1,1), 0.1); 
}

PsInput vs_instanced(VsInputInstanced input)
{
    PsInput output;
    output.pos_o = input.pos.xyz;
    float3 wpos = input.pos.xyz + float3(input.offset.x, input.offset.y, input.offset.z);
    output.pos = mul(g_viewProj, float4(wpos, 1));    
    return output;
}

float4 ps_instanced(PsInput input) : SV_Target
{
    float3 abs_pos = abs(input.pos_o.xyz);
    float edge_width = 0.025;
    float3 mask = smoothstep(0.5 - edge_width, 0.5, abs_pos);
    float alpha = max(max(mask.x * mask.y, mask.y * mask.z), mask.z * mask.x);
    alpha = alpha > 0.5 ? 1.0 : 0.5;
    return float4(float3(1,1,1), alpha); 
}
END

compile_shader_d3d :: (source: string, entry_point: string, shader_model: string) -> string, HRESULT {
    flags := D3DCOMPILE.ENABLE_STRICTNESS;
    flags |= .PACK_MATRIX_ROW_MAJOR;
    #if DEBUG {
        flags |= .DEBUG;
        flags |= .SKIP_OPTIMIZATION;
    }

    defines: [] string;
    bytecode, errors, hr := D3DCompile(source, "source", defines, null, entry_point, shader_model, xx flags, 0);
    defer free(errors);

    if FAILED(hr) {
        free(bytecode);
        if errors print ("Error msg: %\n", errors);
        return "", hr;
    }
    return bytecode, hr;
}

compile_shader :: (source: string, vs_entry: string, ps_entry: string,
    vertex_shader: **ID3D11VertexShader, pixel_shader: **ID3D11PixelShader,
    il_desc: []D3D11_INPUT_ELEMENT_DESC, il: **ID3D11InputLayout)
{
    ps : string;
    hr : HRESULT;
    ps, hr = compile_shader_d3d(source, ps_entry, "ps_4_0");
    if FAILED(hr) {
        log_error("Compile PS failed: %", hr);
    }

    hr = ID3D11Device_CreatePixelShader(g_device, ps.data, cast(u64) ps.count, null, pixel_shader);
    if FAILED(hr) {
        log_error("CreatePixelShader failed: %", hr);
    }

    vs : string;
    vs, hr = compile_shader_d3d(source, vs_entry, "vs_4_0");
    if FAILED(hr) {
        log_error("Compile VS failed: %", hr);
    }
    assert(vs.data != null);

    hr = ID3D11Device_CreateVertexShader(g_device, vs.data, cast(u64) vs.count, null, 
        vertex_shader);
    if FAILED(hr) {
        log_error("CreateVertexShader failed: %", hr);
    }
    
    hr = ID3D11Device_CreateInputLayout(g_device, il_desc.data, cast(u32) il_desc.count,
        vs.data, cast(u64) vs.count, il);
    if FAILED(hr) {
        log_error("CreateInputLayout failed: %", hr);
    }

    free(vs);
    free(ps);
}