// SPDX-FileCopyrightText: James Webb
// SPDX-License-Identifier: MIT
// Full details are in the LICENSE file at the repository root.

#import "Basic";
#import "d3d11"()(INCLUDE_DEBUG_BINDINGS = false);
#import "Math";
#import "Random";
#import "Windows";
#import "SDL";
#import "File_Utilities";
#import "Windows_Utf8";
#import "dxgi";
ImGui :: #import "ImGui";
#load "src/imgui_impl_sdl_dx11.jai";
#load "src/camera.jai";
#load "src/rendering.jai";

DEBUG :: 0;

AppState :: struct
{
    total_time         : float = 0;
    last_update_time   : float = 0;

    nn                 : NeuralNet;
    nn_instance        : NeuralNetInstance;
    training_config    : TrainingConfig;
    test_data          : [..]TestData;
    training_data      : [..]TestData;
    num_run_tests      : u32 = 0;
    num_passed_tests   : u32 = 0;
    update_interval    : float = 2.5;

    network_files          : [..]string;
    selected_network_file  : s32 = 0;
    output_filename_buffer : [256]u8;

    force_update         : bool = true;
    current_output_idx   : u8;
    active_test_idx      : u32 = 0;
    last_test_idx        : u32 = 1;
    mnist_output_labels  : [MNIST_MAPPING.count][2]u8;
    emnist_output_labels : [EMNIST_BALANCED_MAPPING.count][2]u8;
}

init_window :: (sdl_window: **SDL_Window, app_state: *AppState) -> bool
{
    SDL_Init(SDL_INIT_VIDEO);
    WINDOW_INIT_WIDTH :: 1650;
    WINDOW_INIT_HEIGHT :: 1024;
    sdl_window.* = SDL_CreateWindow("Jai Machine Learning", SDL_WINDOWPOS_CENTERED, 
        SDL_WINDOWPOS_CENTERED, WINDOW_INIT_WIDTH, WINDOW_INIT_HEIGHT, 0);
    if sdl_window.* == null {
        log_error("Could not create window: %", to_string(SDL_GetError()));
        exit(1);
    }
    return true;
}

init_ml :: (app_state: *AppState, train: bool = true)
{
    init_nn(*app_state.nn, *app_state.training_config);
    load_data(app_state.training_config.data_set, *app_state.training_data, *app_state.test_data);
    if (train) {
        instance : NeuralNetInstance;
        init_nn_instance(*app_state.nn, *instance);
        train_network(*app_state.nn, *instance, app_state.training_data, app_state.training_config);
    }
    init_nn_instance(*app_state.nn, *app_state.nn_instance);
    set_inputs(*app_state.nn, *app_state.nn_instance, app_state.test_data[app_state.active_test_idx].pixels);
}

read_from_mem :: (src: *u8, src_offset: *u32, dst: *void, size: s64)
{
    memcpy(dst, src + src_offset.*, size);
    src_offset.* += xx size;
}

clear_test_history :: (app_state: *AppState)
{    
    app_state.num_run_tests = 0;
    app_state.num_passed_tests = 0;
}

load_network_from_file :: (app_state: *AppState, render_state: *RenderState, file: string) -> bool
{
    data, success := read_entire_file(file);
    if !success {
        log_error("Unable to read file '%'.\n", file);
        return false;
    }
    defer free(data);

    offset : u32 = 0;
    read_from_mem(data.data, *offset, *app_state.nn.loss, size_of(u32));
    read_from_mem(data.data, *offset, *app_state.training_config.data_set, size_of(u32));
    read_from_mem(data.data, *offset, *app_state.training_config.loss_func, size_of(u32));
    read_from_mem(data.data, *offset, render_state.layer_brightness_scale.data, 8 * size_of(u32));
    read_from_mem(data.data, *offset, *render_state.camera.position.x, size_of(Vector3));
    read_from_mem(data.data, *offset, *render_state.camera.euler_angles.x, size_of(Vector3));
    read_from_mem(data.data, *offset, *render_state.active_gradient, size_of(u32));
    read_from_mem(data.data, *offset, *render_state.bias, size_of(u32));
    num_hidden : u32 = 0;
    read_from_mem(data.data, *offset, *num_hidden, size_of(u32));
    array_resize(*app_state.training_config.hidden_layers, num_hidden);
    read_from_mem(data.data, *offset, app_state.training_config.hidden_layers.data, num_hidden * size_of(LayerDesc));
    
    configure_layers(*app_state.training_config);
    init_ml(app_state, false);

    for l:1..app_state.nn.layers.count-1 {
        for i:0..app_state.nn.layers[l].weights.count-1 {
            read_from_mem(data.data, *offset, app_state.nn.layers[l].weights[i].data, app_state.nn.layers[l].weights[i].count * size_of(float));
        }
    }
    for l:1..app_state.nn.layers.count-1 {
        read_from_mem(data.data, *offset, app_state.nn.layers[l].biases.data, app_state.nn.layers[l].biases.count * size_of(float));
    }
    assert(offset == data.count);

    clear_test_history(app_state);
    init_instance_positions(*app_state.nn, render_state);
    memcpy(app_state.output_filename_buffer.data, app_state.network_files[app_state.selected_network_file].data,
        app_state.network_files[app_state.selected_network_file].count);
    app_state.output_filename_buffer.data[app_state.network_files[app_state.selected_network_file].count] = 0;

    return true;
}

write_to_mem :: (data: []u8, offset: *u32, src: *void, size: s64)
{
    memcpy(data.data + offset.*, src, size);
    offset.* += xx size;
}

save_network_to_file :: (app_state: *AppState, render_state: *RenderState, file: string) -> bool
{
    layer_count := app_state.training_config.hidden_layers.count;
    desc_bytes := (20 * size_of(u32)) + (size_of(LayerDesc) * layer_count);
    
    total_weights := 0;
    total_biases := 0;
    for l:1..app_state.nn.layers.count-1 {
        total_weights += app_state.nn.layers[l].weights.count * app_state.nn.layers[l].weights[0].count;
        total_biases += app_state.nn.layers[l].biases.count;
    }
    num_bytes := desc_bytes + (total_biases * size_of(float)) + (total_weights * size_of(float));

    data : [..]u8;
    array_resize(*data, num_bytes);
    defer array_free(data);
    offset : u32 = 0;
    write_to_mem(data, *offset, *app_state.nn.loss, size_of(u32));
    write_to_mem(data, *offset, *app_state.training_config.data_set, size_of(u32));
    write_to_mem(data, *offset, *app_state.training_config.loss_func, size_of(u32));
    write_to_mem(data, *offset, render_state.layer_brightness_scale.data, 8 * size_of(u32));
    write_to_mem(data, *offset, *render_state.camera.position.x, size_of(Vector3));
    write_to_mem(data, *offset, *render_state.camera.euler_angles.x, size_of(Vector3));
    write_to_mem(data, *offset, *render_state.active_gradient, size_of(u32));
    write_to_mem(data, *offset, *render_state.bias, size_of(u32));
    count : u32 = xx app_state.training_config.hidden_layers.count;
    write_to_mem(data, *offset, *count, size_of(u32));
    for i:0..app_state.training_config.hidden_layers.count-1 {
        write_to_mem(data, *offset, *app_state.training_config.hidden_layers[i], size_of(LayerDesc));
    }
    
    for l:1..app_state.nn.layers.count-1 {
        for i:0..app_state.nn.layers[l].weights.count-1 {
            write_to_mem(data, *offset, app_state.nn.layers[l].weights[i].data, app_state.nn.layers[l].weights[i].count * size_of(float));
        }
    }
    for l:1..app_state.nn.layers.count-1 {
        write_to_mem(data, *offset, app_state.nn.layers[l].biases.data, app_state.nn.layers[l].biases.count * size_of(float));
    }
    assert(offset == num_bytes);

    return write_entire_file(file, data.data, data.count);
}

update :: (app_state: *AppState, render_state: *RenderState)
{
    app_state.total_time += ImGui.GetIO().DeltaTime;
    
    // @TODO: could support multiple updates in here to handle above framerate iteration
    if (app_state.force_update || app_state.total_time - app_state.last_update_time > app_state.update_interval) {
        app_state.active_test_idx += 1;
        app_state.active_test_idx = app_state.active_test_idx % xx app_state.test_data.count;
        set_inputs(*app_state.nn, *app_state.nn_instance, app_state.test_data[app_state.active_test_idx].pixels);
        app_state.total_time = app_state.last_update_time;
        propagate_forward(*app_state.nn, *app_state.nn_instance);
    }
    if (app_state.force_update || app_state.last_test_idx != app_state.active_test_idx) {
        app_state.num_run_tests += 1;
        app_state.last_test_idx = app_state.active_test_idx;
        output_activations := app_state.nn_instance.layer_states[app_state.nn_instance.layer_states.count - 1].activation;
        max_value := -1.0;
        max_idx := 0;
        for i:0..output_activations.count-1 {
            if (output_activations[i] > max_value) {
                max_idx = i;
                max_value = output_activations[i];
            }
        }
        ref_value := app_state.test_data[app_state.active_test_idx].value;
        app_state.current_output_idx = xx max_idx;
        if max_idx == ref_value {
            app_state.num_passed_tests += 1;
        }
        render_state.buffers_dirty = true;
    }
    app_state.force_update = false;
}

render :: (app_state: *AppState, render_state: *RenderState)
{
    update_camera(*render_state.camera, ImGui.GetIO().DeltaTime, render_state.window_width, render_state.window_height);

    for i:0..render_state.output_neuron_positions.count-1 {
        pos := render_state.output_neuron_positions[i];
        proj := project_screen(pos, render_state.camera.view_projection_matrix, xx render_state.window_width, xx render_state.window_height);
        draw_list := ImGui.GetForegroundDrawList();
        if proj.z > 0 {
            impos : ImGui.ImVec2 = .{ proj.x - 5, proj.y - 5 };
            if (app_state.training_config.data_set == .EMNIST_Balanced) {
                ImGui.ImDrawList.AddText(draw_list, *impos, 0xFFFFFFFF, app_state.emnist_output_labels[i].data, app_state.emnist_output_labels[i].data + 1);
            }
            else {
                ImGui.ImDrawList.AddText(draw_list, *impos, 0xFFFFFFFF, app_state.mnist_output_labels[i].data, app_state.mnist_output_labels[i].data + 1);
            }
        }
    }

    ID3D11DeviceContext_OMSetRenderTargets(render_state.ctx, 1, *render_state.rtv, null);
    vp : D3D11_VIEWPORT;
    vp.Width = xx (render_state.window_width * render_state.res_scaling);
    vp.Height = xx (render_state.window_height * render_state.res_scaling);
    vp.MinDepth = 0.0;
    vp.MaxDepth = 1.0;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    ID3D11DeviceContext_RSSetViewports(render_state.ctx, 1, *vp);

    clear_color: [4] float = .[ 26.0/255.0, 29.0/255.0, 43.0/255.0, 1 ];
    ID3D11DeviceContext_ClearRenderTargetView(render_state.ctx, render_state.rtv, *clear_color);

    render_state.frame_idx += 1;
    if (render_state.buffers_dirty) {
        update_neuron_buffers(render_state, render_state.camera.position, *app_state.nn, *app_state.nn_instance);
        render_state.wire_vertices = update_wire_buffers(render_state, *app_state.nn, *app_state.nn_instance);
        render_state.buffers_dirty = false;
    }

    cb: ShaderConstants;
    cb.mvp = render_state.camera.view_projection_matrix;
    cb.bias_time_unused.x = render_state.bias;
    cb.bias_time_unused.y = app_state.total_time;
    ID3D11DeviceContext_UpdateSubresource(render_state.ctx, render_state.cbuffer, 0, null, *cb, 0, 0);

    ID3D11DeviceContext_RSSetState(render_state.ctx, render_state.rs);
    ID3D11DeviceContext_OMSetBlendState(render_state.ctx, render_state.bs, null, 0xffffffff);
    cbuffers: [1] *ID3D11Buffer;
    cbuffers[0] = render_state.cbuffer;
    ID3D11DeviceContext_VSSetConstantBuffers(render_state.ctx, 0, 1, *cbuffers[0]);
    ID3D11DeviceContext_PSSetConstantBuffers(render_state.ctx, 0, 1, *cbuffers[0]);

    srvs: [2] *ID3D11ShaderResourceView;
    srvs[0] = null;
    srvs[1] = render_state.gradient_texture_srvs[render_state.active_gradient];
    ID3D11DeviceContext_PSSetShaderResources(render_state.ctx, 0, 2, *srvs[0]);
    samplers: [1] *ID3D11SamplerState;
    samplers[0] = render_state.sampler;
    ID3D11DeviceContext_PSSetSamplers(render_state.ctx, 0, 1, *samplers[0]);

    // Draw neurons
    ID3D11DeviceContext_VSSetShader(render_state.ctx, render_state.vs_instanced, null, 0);
    ID3D11DeviceContext_PSSetShader(render_state.ctx, render_state.ps_instanced, null, 0);
    stride: u32 = size_of(Vector3);
    offset: u32 = 0;
    ID3D11DeviceContext_IASetInputLayout(render_state.ctx, render_state.il_instanced);
    ID3D11DeviceContext_IASetVertexBuffers(render_state.ctx, 0, 1, *render_state.vb_cube, *stride, *offset);
    stride = size_of(InstanceVertex);
    ID3D11DeviceContext_IASetVertexBuffers(render_state.ctx, 1, 1, *render_state.vb_neuron_inst, *stride, *offset);
    ID3D11DeviceContext_IASetPrimitiveTopology(render_state.ctx, D3D11_PRIMITIVE_TOPOLOGY.TRIANGLELIST);
    ID3D11DeviceContext_DrawInstanced(render_state.ctx, 36, xx render_state.neuron_vertex_instances.count, 0, 0);

    // Draw wires
    ID3D11DeviceContext_VSSetShader(render_state.ctx, render_state.vs, null, 0);
    ID3D11DeviceContext_PSSetShader(render_state.ctx, render_state.ps, null, 0);
    ID3D11DeviceContext_IASetInputLayout(render_state.ctx, render_state.il);
    ID3D11DeviceContext_IASetVertexBuffers(render_state.ctx, 0, 1, *render_state.vb_wire_inst, *stride, *offset);
    ID3D11DeviceContext_IASetPrimitiveTopology(render_state.ctx, D3D11_PRIMITIVE_TOPOLOGY.LINELIST);
    ID3D11DeviceContext_Draw(render_state.ctx, xx render_state.wire_vertices, 0);    
    // print ("Draw % of possible %\n", wire_vertices, render_state.wire_vertex_data.count);

    vp.Width = xx render_state.window_width;
    vp.Height = xx render_state.window_height;
    ID3D11DeviceContext_RSSetViewports(render_state.ctx, 1, *vp);
    ID3D11DeviceContext_IASetInputLayout(render_state.ctx, null);
    ID3D11DeviceContext_IASetPrimitiveTopology(render_state.ctx, D3D11_PRIMITIVE_TOPOLOGY.TRIANGLELIST);
    ID3D11DeviceContext_OMSetRenderTargets(render_state.ctx, 1, *render_state.swap_rtv, null);
    ID3D11DeviceContext_VSSetShader(render_state.ctx, render_state.vs_fullscreen, null, 0);
    ID3D11DeviceContext_PSSetShader(render_state.ctx, render_state.ps_fullscreen, null, 0);

    srvs[0] = render_state.srv;
    srvs[1] = null;
    ID3D11DeviceContext_PSSetShaderResources(render_state.ctx, 0, 2, *srvs[0]);
    ID3D11DeviceContext_Draw(render_state.ctx, 3, 0);
}

build_files_list :: (info : *File_Visit_Info, app_state : *AppState)
{
    path, basename, ext := path_decomp (info.full_name);
    if ext == "nn" {
        copy := alloc_string(basename.count);
        memcpy(copy.data, basename.data, basename.count);
        copy.data[copy.count] = 0;
        array_add(*app_state.network_files, copy);
    }
}

draw_ui :: (app_state: *AppState, render_state: *RenderState)
{
    ImGui.Begin("Network");
    training_config := *app_state.training_config;
    layer_count: s32 = xx training_config.hidden_layers.count;
    data_sets_str :: "MNIST\0EMNIST\0";
    ImGui.Text("Dataset");
    ImGui.SameLine();
    ImGui.SetNextItemWidth(208);
    curr_dataset := cast(s32)app_state.training_config.data_set;
    if (ImGui.Combo("##dataset", *curr_dataset, data_sets_str.data, 2)) {
        app_state.training_config.data_set = xx curr_dataset;
    }        
    loss_funcs_str :: "MSE\0Cross entropy\0";
    ImGui.Text("Loss calculation");
    ImGui.SameLine();
    ImGui.SetNextItemWidth(145);
    curr_lossfunc := cast(s32)app_state.training_config.loss_func;
    if (ImGui.Combo("##lossfunc", *curr_lossfunc, loss_funcs_str.data, 2)) {
        app_state.training_config.loss_func = xx curr_lossfunc;
    }

    ImGui.Text("Layers ");
    ImGui.SameLine();
    ImGui.SetNextItemWidth(208);
    if ImGui.DragInt("##layers", *layer_count, 1.0, 2, 5) {
        old_layer_count: s32 = xx training_config.hidden_layers.count;
        array_resize(*training_config.hidden_layers, xx layer_count);
        for i:0..training_config.hidden_layers.count-1 {
            if (training_config.hidden_layers[i].size < 4 || training_config.hidden_layers[i].size > 256) {
                training_config.hidden_layers[i].size = 4;
                training_config.hidden_layers[i].activation = .ReLU;
            }
        }
    }
    layer_sizes_str :: "4\09\016\025\036\049\064\0100\0144\0256\0400\0625\01024\0";
    layer_sizes :: u32.[4, 9, 16, 25, 36, 49, 64, 100, 144, 256, 400, 625, 1024];
    layer_activation_str :: "Linear\0Sigmoid\0ReLU\0";
    for i:0..training_config.hidden_layers.count-1 {
        push_allocator(temp);
        formatted_text_builder: String_Builder;
        print_to_builder(*formatted_text_builder, "Layer %", i);
        append(*formatted_text_builder, "\0");
        formatted_text := builder_to_string(*formatted_text_builder);
        found:, idx: = array_find(layer_sizes, training_config.hidden_layers[i].size);
        idx_s32 : s32 = xx idx;
        assert(found, "Invalid layer size");
        ImGui.PushID(cast(s32)i);
        ImGui.Text_CFormat("%s", formatted_text.data);
        ImGui.SameLine();
        ImGui.SetNextItemWidth(100);
        if (ImGui.Combo("##layer_size", *idx_s32, layer_sizes_str.data, layer_sizes.count)) {
            training_config.hidden_layers[i].size = layer_sizes[idx_s32];
        }
        ImGui.SameLine();
        ImGui.SetNextItemWidth(100);
        curr_active: s32 = xx training_config.hidden_layers[i].activation;
        if (ImGui.Combo("##layer_activation", *curr_active, layer_activation_str.data, 5)) {                
            training_config.hidden_layers[i].activation = xx curr_active;
        }
        ImGui.PopID();
    }
    
    ImGui.Text("Learning Rate");
    ImGui.SameLine();
    ImGui.SetNextItemWidth(166);
    ImGui.DragFloat("##learn_rate", *training_config.learning_rate, 0.01, 0.001, 0.1);

    ImGui.Text("Epochs");
    ImGui.SameLine();
    ImGui.SetNextItemWidth(215);
    epoch_count: s32 = xx training_config.num_epochs;
    if ImGui.DragInt("##epochs", *epoch_count, 1.0, 1, 100) {
        training_config.num_epochs = xx epoch_count;
    }
    if ImGui.Button("Rebuild and train") {
        configure_layers(*app_state.training_config);
        init_ml(app_state);
        init_instance_positions(*app_state.nn, render_state);
    }
    ImGui.SameLine();
    if ImGui.Button("Continue training") {            
        train_network(*app_state.nn, *app_state.nn_instance, app_state.training_data, app_state.training_config);
    }
    if ImGui.Button("Save As") {
        out_name : string;
        out_name.count = c_style_strlen(app_state.output_filename_buffer.data);
        out_name.data = app_state.output_filename_buffer.data;
        full_path := tprint("data/%.nn", out_name);
        save_network_to_file(app_state, render_state, full_path);
        array_reset(*app_state.network_files);
        app_state.selected_network_file = 0;
        visit_files("data", false, app_state, build_files_list);
    }
    ImGui.SameLine();
    ImGui.SetNextItemWidth(200);
    ImGui.InputText("##filename", app_state.output_filename_buffer.data, 256);

    ImGui.Text("Load    ");
    ImGui.SameLine();
    ImGui.SetNextItemWidth(200);
    if (app_state.network_files.count > 0) {
        preview_text := ifx app_state.selected_network_file >= 0 && app_state.selected_network_file < app_state.network_files.count
            then app_state.network_files[app_state.selected_network_file]
            else "None";
        if ImGui.BeginCombo("##file", preview_text.data) {
            for i:0..app_state.network_files.count-1 {
                selected := app_state.selected_network_file == i;
                if ImGui.Selectable(app_state.network_files[i].data, selected) {
                    app_state.selected_network_file = xx i;
                    full_path := tprint("data/%.nn", app_state.network_files[i]);
                    load_network_from_file(app_state, render_state, full_path);
                    app_state.force_update = true;
                }
                if (selected) {
                    ImGui.SetItemDefaultFocus();
                }
            }
            ImGui.EndCombo();
        }
    }
    else {
        if ImGui.BeginCombo("##file", "None") {                
            ImGui.EndCombo();
        }
    }
    ImGui.End();

    ImGui.Begin("Results");
    data_set := ifx (app_state.training_config.data_set == .EMNIST_Balanced) then EMNIST_BALANCED_MAPPING.data else MNIST_MAPPING.data;
    actual_value := data_set[app_state.current_output_idx];
    expected_value := data_set[app_state.test_data[app_state.active_test_idx].value];
    output_colour: ImGui.ImVec4 = .{ 1.0, 0, 0, 1.0 };
    if actual_value == expected_value {
        output_colour = .{ 0.0, 1.0, 0, 1.0 };
    }
    ImGui.TextColored_CFormat(*output_colour, "Current output: %s", u8.[actual_value, 0].data);
    ImGui.TextColored_CFormat(*output_colour, "Current expected output: %s", u8.[expected_value, 0].data);

    ImGui.Text("Tests run: %", app_state.num_run_tests);
    ImGui.SameLine();
    if ImGui.Button("Clear") {
        clear_test_history(app_state);
    }
    ImGui.Text("Tests passed: %", app_state.num_passed_tests);
    ImGui.Text("Loss: %", app_state.nn.loss);
    ImGui.Text("Accuracy: %\%", cast(float)app_state.num_passed_tests / cast(float)app_state.num_run_tests * 100);
    ImGui.Text("Update interval");
    ImGui.SameLine();
    ImGui.SetNextItemWidth(200);
    ImGui.SliderFloat("##update_interval", *app_state.update_interval, 0.001, 100.0, "%.3f", .Logarithmic);
    ImGui.End();
    
    ImGui.Begin("Visualisation");
    ImGui.Text("FPS: %", 1.0 / ImGui.GetIO().DeltaTime);
    ImGui.SetNextItemWidth(200);
    ImGui.DragFloat4("Layer brightness", xx *render_state.layer_brightness_scale[0], 0.01);
    ImGui.SetNextItemWidth(200);
    ImGui.DragFloat4("##brightness2", xx *render_state.layer_brightness_scale[4], 0.01);
    ImGui.SliderFloat("Activation draw threshold", *render_state.activation_draw_threshold, 0.01, 0.99, "%.2f");
            
    themes_str :: "Red/blue\0House of cards\0Grand\0Yellow/green/blue\0Neo\0Blue/white\0White\0";
    ImGui.Text("Theme");
    ImGui.SameLine();
    ImGui.SetNextItemWidth(200);
    ImGui.Combo("##theme", *render_state.active_gradient, themes_str.data, 10);
    ImGui.SetNextItemWidth(200);
    ImGui.SliderFloat("Colour bias", *render_state.bias, 0.0, 1.0);
    ImGui.End();
}

#scope_export
main :: ()
{
    sdl_window   : *SDL_Window;
    render_state : RenderState;
    app_state    : AppState;
    imgui_state  : ImGuiState;

    DEFAULT_FILE_NAME :: "default.nn";
    memcpy(app_state.output_filename_buffer.data, DEFAULT_FILE_NAME.data, DEFAULT_FILE_NAME.count);
    app_state.output_filename_buffer[DEFAULT_FILE_NAME.count] = 0;

    if !init_window(*sdl_window, *app_state) {
        print ("init_window failed!\n");
        return;
    }
    wmInfo: SDL_SysWMinfo;
    SDL_VERSION(*wmInfo.version);
    SDL_GetWindowWMInfo(sdl_window, *wmInfo);
    render_state.window_hwnd = wmInfo.info.win.window;

	SDL_GetWindowSize(sdl_window, *render_state.window_width, *render_state.window_height);
    if !init_device(*render_state) {
        print ("init_device failed!\n");
        cleanup_device(*render_state);
        return;
    }

    imgui_state.device = render_state.device;
    imgui_state.ctx = render_state.ctx;
    imgui_context := ImGui.CreateContext();
    ImGui_ImplSdl_Init(*imgui_state, sdl_window);

    render_state.camera.position = .{ -50, 14 * 1.5, 55 };
    render_state.camera.euler_angles.x = PI/2;

    visit_files("data", false, *app_state, build_files_list);
    if app_state.network_files.count > 0 {
        full_path := tprint("data/%.nn", app_state.network_files[app_state.selected_network_file]);
        load_network_from_file(*app_state, *render_state, full_path);
    }
    else {
        configure_layers_default(*app_state.training_config);
        init_ml(*app_state);
        init_instance_positions(*app_state.nn, *render_state);
    }

    for i:0..app_state.mnist_output_labels.count-1 {
        app_state.mnist_output_labels[i][0] = MNIST_MAPPING[i];
        app_state.mnist_output_labels[i][1] = 0;
    }    
    for i:0..app_state.emnist_output_labels.count-1 {
        app_state.emnist_output_labels[i][0] = EMNIST_BALANCED_MAPPING[i];
        app_state.emnist_output_labels[i][1] = 0;
    }

    done := false;
    while !done {
        reset_temporary_storage();

        event: SDL_Event;
        while SDL_PollEvent(*event) {
            ImGui_ImplSdl_ProcessEvent(*imgui_state, *event);
            if event.type == SDL_QUIT {
                done = true;
            }
            if event.type == SDL_WINDOWEVENT && (event.window.event == SDL_WINDOWEVENT_CLOSE)
                && (event.window.windowID == SDL_GetWindowID(sdl_window)) {
                done = true;
            }
        }
        ImGui_ImplSdl_NewFrame(*imgui_state, sdl_window);
        ImGui.NewFrame();
        
        if ImGui.GetMouseCursor() == {
            case .Arrow;         SDL_SetCursor(cursor_arrow);
            case .TextInput;     SDL_SetCursor(cursor_text_input);
            case .Hand;          SDL_SetCursor(cursor_move);
            case .ResizeNS;      SDL_SetCursor(cursor_resize_ns);
            case .ResizeEW;      SDL_SetCursor(cursor_resize_ew);
            case .ResizeNESW;    SDL_SetCursor(cursor_resize_nesw);
            case .ResizeNWSE;    SDL_SetCursor(cursor_resize_nwse);
        }

        draw_ui(*app_state, *render_state);
        update(*app_state, *render_state);
        render(*app_state, *render_state);

        ImGui.Render();
        draw_data := ImGui.GetDrawData();
        ImGui_ImplSdl_RenderDrawLists(*imgui_state, draw_data);
        
        hr := IDXGISwapChain_Present(render_state.swap_chain, 0, 0);
        if FAILED(hr) {
            log_error("Present failed: %", hr);
        }
    }

    ImGui_ImplSdl_Shutdown(*imgui_state);
    ImGui.DestroyContext(imgui_context);
    cleanup_device(*render_state);
    SDL_DestroyWindow(sdl_window);
    SDL_Quit();
}
