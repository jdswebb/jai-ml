// SPDX-FileCopyrightText: James Webb
// SPDX-License-Identifier: MIT
// Full details are in the LICENSE file at the repository root.

#load "mappings.jai";

Layer :: struct
{
    size            : u32;
    weights         : [..][..]float;
    biases          : [..]float;
    activation_type : ActivationType;
}

LayerState :: struct
{
    pre_activation : [..]float;
    activation     : [..]float;
    min_activation : float;
    max_activation : float;
}

NeuralNet :: struct
{
    layers : [..]Layer;
    loss : float;
}

NeuralNetInstance :: struct
{
    net          : *NeuralNet;
    layer_states : [..]LayerState;
}

LayerDesc :: struct
{
    size       : u32;
    activation : ActivationType;
}

DataSet :: enum u32
{
    MNIST;
    EMNIST_Balanced;
}

LossFunction :: enum u32
{
    MSE;
    CrossEntropy;
}

TrainingConfig :: struct
{
    IMAGE_SIZE    :: 28;
    data_set      : DataSet = .EMNIST_Balanced;
    loss_func     : LossFunction = .MSE;
    num_epochs    : u32 = 1;
    learning_rate : float = 0.05;
    hidden_layers : [..]LayerDesc;
    layers        : [..]LayerDesc;
}

LayerGradients :: struct
{
    weights : [..][..]float;
    biases  : [..]float;
}

TrainingData :: struct
{
    net: *NeuralNet;
    layer_deltas: [..][..]float;
    layer_gradients: [..]LayerGradients;
}

ActivationType :: enum u8
{
    Linear   :: 0;
    Sigmoid  :: 1;
    ReLU     :: 2;
}

TestData :: struct
{
    // no intention to support anything being (E)MNIST for now
    value: u8;
    pixels: [TrainingConfig.IMAGE_SIZE * TrainingConfig.IMAGE_SIZE]u8;
}

activation :: inline (type: ActivationType, x: float) -> float
{
    if #complete type == {
        case .Linear;
            return linear(x);
        case .Sigmoid;
            return sigmoid(x);
        case .ReLU;
            return relu(x);
    };
}

activation_deriv :: inline (type: ActivationType, x: float) -> float
{
    if #complete type == {
        case .Linear;
            return linear_deriv(x);
        case .Sigmoid;
            return sigmoid_deriv(x);
        case .ReLU;
            return relu_deriv(x);
    };
}

sigmoid :: inline (x: float) -> float
{
    return 1.0 / (1.0 + exp(-x));
}

sigmoid_deriv :: inline (x: float) -> float
{
    s := sigmoid(x);
    return s * (1.0 - s);
}

linear :: inline (x: float) -> float
{
    return x;
}

linear_deriv :: inline (x: float) -> float
{
    return 1;
}

relu :: inline (x: float) -> float
{
    return max(0.0, x);
}

relu_deriv :: inline (x: float) -> float
{
    if x > 0.0 
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

softmax :: (logits: []float, out_probs: []float)
{
    assert(logits.count == out_probs.count);
    max_val := logits[0];
    for i:1..logits.count-1 {
        if logits[i] > max_val {
            max_val = logits[i];
        }
    }

    sum := 0.0;
    for i:0..logits.count-1 {
        out_probs[i] = exp(logits[i] - max_val);
        sum += out_probs[i];
    }

    for i:0..logits.count-1 {
        out_probs[i] /= sum;
    }
}

sample_normal :: (mean: float, stddev: float) -> float
{
    u1 := random_get_within_range(0.0000001, 1.0);
    u2 := random_get_within_range(0.0, 1.0);
    z0 := sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    return z0 * stddev + mean;
}

init_nn :: (nn: *NeuralNet, training_config: *TrainingConfig)
{
    array_resize(*nn.layers, training_config.layers.count);
    for l:0..nn.layers.count-1 {
        nn.layers[l].size = training_config.layers[l].size;
    }
    for l:1..nn.layers.count-1 {
        nn.layers[l].activation_type = training_config.layers[l].activation;        
        array_resize(*nn.layers[l].weights, training_config.layers[l].size);
        array_resize(*nn.layers[l].biases, training_config.layers[l].size);
        for j:0..nn.layers[l].size-1 {
            array_resize(*nn.layers[l].weights[j], training_config.layers[l - 1].size);
        }
    }
    for l:1..nn.layers.count-1 {
        for i:0..nn.layers[l].size-1 {
            nn.layers[l].biases[i] = 0.01;
        }
    }
    randomise_weights(nn);
}

randomise_weights :: (nn: *NeuralNet)
{
    for l:1..nn.layers.count-1 {
        limit := 0.1;
        log("Using % for layer %", nn.layers[l].activation_type, l);
        if (nn.layers[l].activation_type == .Sigmoid) || (nn.layers[l].activation_type == .Linear) {
            // Xavier Initialization 
            fan_in  := nn.layers[l-1].size;
            fan_out := nn.layers[l].size;
            limit = sqrt(6.0 / (fan_in + fan_out));
        }
        if (nn.layers[l].activation_type == .Linear) {
            // Xavier Initialization 
            fan_in  := nn.layers[l-1].size;
            fan_out := nn.layers[l].size;
            limit = sqrt(6.0 / (fan_in + fan_out));
        }
        else {
            // He initialization for ReLU
            fan_in := nn.layers[l-1].size;
            limit = sqrt(2.0 / fan_in);
        }
        for i:0..nn.layers[l].size-1 {
            for j:0..nn.layers[l-1].size-1 {
                nn.layers[l].weights[i][j] = random_get_within_range(-limit, limit);
                // nn.layers[l].weights[i][j] = sample_normal(0.0, 2.0 / (nn.layers[l].size + nn.layers[l - 1].size));
            }
        }
    }
}

init_nn_instance :: (nn: *NeuralNet, instance: *NeuralNetInstance)
{
    instance.net = nn;
    array_resize(*instance.layer_states, nn.layers.count);
    for _, l:nn.layers {
        array_resize(*instance.layer_states[l].activation, nn.layers[l].size);
        array_resize(*instance.layer_states[l].pre_activation, nn.layers[l].size);
        for v:0..nn.layers[l].size-1 {
            instance.layer_states[l].activation[v] = 1.0;
        }
    }
}

init_training :: (nn: *NeuralNet, td: *TrainingData)
{
    td.net = nn;
    array_resize(*td.layer_deltas, nn.layers.count);
    array_resize(*td.layer_gradients, nn.layers.count);
    for l:1..nn.layers.count-1 {
        sz := nn.layers[l].size;
        array_resize(*td.layer_deltas[l], sz);
        array_resize(*td.layer_gradients[l].weights, sz);
        array_resize(*td.layer_gradients[l].biases, sz);
        for i:0..sz-1 {
            array_resize(*td.layer_gradients[l].weights[i], nn.layers[l - 1].size);
        }
    }
}

set_inputs :: (net: *NeuralNet, instance: *NeuralNetInstance, pixels: []u8)
{
    state := *instance.layer_states[0];
    state.min_activation = FLOAT32_MAX;
    state.max_activation = -FLOAT32_MAX;
    for i:0..net.layers[0].size-1 {
        state.activation[i] = pixels[i] / 255.0;
        state.min_activation = min(state.min_activation, state.activation[i]);
        state.max_activation = max(state.max_activation, state.activation[i]);
    }
}

propagate_forward :: (nn: *NeuralNet, instance: *NeuralNetInstance)
{
    assert(instance.layer_states.count == nn.layers.count);
    for l:1..nn.layers.count-1 {
        layer      := *nn.layers[l];
        curr_size  := nn.layers[l].size;
        state      := *instance.layer_states[l];
        prev_state := *instance.layer_states[l - 1];
        prev_size  := nn.layers[l-1].size;
        i := 0;
        
        state.min_activation = FLOAT32_MAX;
        state.max_activation = -FLOAT32_MAX;
        while i < curr_size {
            partials : [8]float;
            j := 0;
            weights := *layer.weights[i].data[0];
            activations := *prev_state.activation.data[0];
            
            // @NOTE: VERY_OPTIMIZED does a much better job on vectorising this, although even
            // with -march=native, no AVX instructions seem to get emitted.
            while j + 7 < prev_size {
                partials[0] += weights[j + 0] * activations[j + 0];
                partials[1] += weights[j + 1] * activations[j + 1];
                partials[2] += weights[j + 2] * activations[j + 2];
                partials[3] += weights[j + 3] * activations[j + 3];
                partials[4] += weights[j + 4] * activations[j + 4];
                partials[5] += weights[j + 5] * activations[j + 5];
                partials[6] += weights[j + 6] * activations[j + 6];
                partials[7] += weights[j + 7] * activations[j + 7];
                j += 8;
            }
            sum := partials[0] + partials[1] + partials[2] + partials[3] + partials[4] + partials[5] + partials[6] + partials[7];
            while j < prev_size {
                sum += weights[j] * prev_state.activation[j];
                j += 1;
            }
            state.pre_activation[i] = sum + layer.biases[i];
            state.activation[i] = activation(layer.activation_type, state.pre_activation[i]);
            // store the min/max to help with renderering - it is not something normally computed
            state.min_activation = min(state.min_activation, state.activation[i]);
            state.max_activation = max(state.max_activation, state.activation[i]);
            i += 1;
        }
    }
}

clear_gradients :: (td: *TrainingData)
{
    for L:0..td.layer_gradients.count-1 {
        for k:0..td.layer_gradients[L].biases.count-1 {
            td.layer_gradients[L].biases[k] = 0;
        }        
        for k:0..td.layer_gradients[L].weights.count-1 {
            for j:0..td.layer_gradients[L].weights[k].count-1 {
                td.layer_gradients[L].weights[k][j] = 0;
            }
        }
    }
}

average_gradients :: (td: *TrainingData, n: float)
{
    for L:0..td.layer_gradients.count-1 {
        for k:0..td.layer_gradients[L].biases.count-1 {
            td.layer_gradients[L].biases[k] /= n;
        }        
        for k:0..td.layer_gradients[L].weights.count-1 {
            for j:0..td.layer_gradients[L].weights[k].count-1 {
                td.layer_gradients[L].weights[k][j] /= n;
            }
        }
    }
}

apply_gradients :: (nn: *NeuralNet, td: *TrainingData, learn_rate: float)
{
    for l:1..nn.layers.count-1 {
        for i:0..nn.layers[l].size-1 {
            nn.layers[l].biases[i] -= learn_rate * td.layer_gradients[l].biases[i];
            for j:0..nn.layers[l-1].size-1 {
                nn.layers[l].weights[i][j] -= learn_rate * td.layer_gradients[l].weights[i][j];
            }
        }
    }
}

propagate_backward :: (nn: *NeuralNet, inst: *NeuralNetInstance, td: *TrainingData, ys: []float, loss_func: LossFunction, curr_value: u8)
{
    // Computing δ(L)_j for all layers
    //    δ(L)_j = ∂C_0/∂a(L)_j * ∂a(L)_j/∂z(L)_j
    // -> δ(L)_j = ∂C_0/∂a(L)_j * σ'(z(L)_j)
    
    // For the output layer, C_0 = sum(a(L)_i-y_i)^2 (MSE - mean squared error)
    // δC_0/∂a(L)_j = ∂/∂a(L)_j [ sum(a(L)_i-y_i)^2 ] = 2(a(L)_j-y_j)
    // because all derivatives in the sum are zero besides when i==j
    output_layer := nn.layers.count - 1;
    if loss_func == .MSE {
        ys[curr_value] = 1.0;
        for j:0..nn.layers[output_layer].size-1 {
            td.layer_deltas[output_layer][j] = 2 * (inst.layer_states[output_layer].activation[j] - ys[j]);
            td.layer_deltas[output_layer][j] *= activation_deriv(nn.layers[output_layer].activation_type, inst.layer_states[output_layer].pre_activation[j]);
        }
    }
    else {
        softmax(inst.layer_states[output_layer].activation, ys);
        for j: 0..nn.layers[output_layer].size {
            td.layer_deltas[output_layer][j] = ys[j];
        }
        td.layer_deltas[output_layer][curr_value] -= 1.0;
    }
    
    // For hidden layer L (i.e., not the output layer), compute the delta as:
    //     δC_0/∂z(L)_k
    //         = sum[j = 0 to N_(L+1) - 1] ( ∂C_0/∂a(L)_k ) * σ'(z(L)_k)
    //         = sum[j = 0 to N_(L+1) - 1] ( δ(L+1)_j * w(L+1)_{j,k} ) * σ'(z(L)_k)
    L := nn.layers.count - 2;
    while L > 0 {
        for j:0..nn.layers[L].size-1 {
            sum: float = 0;
            layer_deltas := td.layer_deltas[L+1];
            for k:0..nn.layers[L+1].size-1 {
                sum += nn.layers[L+1].weights[k][j] * layer_deltas[k];
            }
            sum *= activation_deriv(nn.layers[L].activation_type, inst.layer_states[L].pre_activation[j]);
            td.layer_deltas[L][j] = sum;
        }
        L -= 1;
    }

    // Compute gradient vector
    for L:1..nn.layers.count-1 {
        for k:0..nn.layers[L].size-1 {
            // δC_0/∂b(L)_k
            td.layer_gradients[L].biases[k] += td.layer_deltas[L][k];

            // These locals help perf a lot
            weights := td.layer_gradients[L].weights[k];
            delta := td.layer_deltas[L][k];
            state := inst.layer_states[L-1];
            curr_size := nn.layers[L-1].size-1;
            // δC_0/∂w(L)_jk
            j := 0;
            while (j + 7) < curr_size {
                weights[j + 0] += delta * state.activation[j + 0];
                weights[j + 1] += delta * state.activation[j + 1];
                weights[j + 2] += delta * state.activation[j + 2];
                weights[j + 3] += delta * state.activation[j + 3];
                weights[j + 4] += delta * state.activation[j + 4];
                weights[j + 5] += delta * state.activation[j + 5];
                weights[j + 6] += delta * state.activation[j + 6];
                weights[j + 7] += delta * state.activation[j + 7];
                j += 8;
            }
            while j < curr_size {
                weights[j] += delta * state.activation[j];
                j += 1;
            }
        }
    }
}

train_network :: (nn: *NeuralNet, instance : *NeuralNetInstance, training_data: [..]TestData, cfg: TrainingConfig) -> float
{
    print ("Start training\n");
    
    indices : [..]s64;
    array_resize(*indices, training_data.count);
    for i:0..training_data.count-1 {
        indices[i] = i;
    }

    shuffle :: (arr: []int) {
        for i : 0..arr.count-1 {
            j := cast(u64) (random_get_zero_to_one() * (arr.count - 1));
            tmp := arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    td : TrainingData;
    init_training(nn, *td);

    batch_size := 128;
    num_batches := (training_data.count / batch_size);

    normalized_loss : float = 0.0;
    ys : [..]float;
    array_resize(*ys, nn.layers[nn.layers.count-1].size);
    for epoch:0..cfg.num_epochs-1 {
        start: Apollo_Time;
        start = current_time_consensus();
        for b:0..num_batches-1 {
            clear_gradients(*td);

            for i:0..batch_size-1 {
                training_idx := (b * batch_size) + i;
                for p:0..nn.layers[0].size-1 {
                    instance.layer_states[0].activation[p] = training_data[indices[training_idx]].pixels[p] / 255.0;
                }
                propagate_forward(nn, instance);
                // backwards pass accumulate, ys = expected value
                memset(ys.data, 0, ys.count * size_of(float));
                propagate_backward(nn, instance, *td, ys, cfg.loss_func, training_data[indices[training_idx]].value);
            }

            // apply gradients
            average_gradients(*td, xx batch_size);
            apply_gradients(nn, *td, cfg.learning_rate);
        }
        shuffle(indices);

        // calculate loss
        loss : float = 0.0;
        for i:0..training_data.count-1 {
            for p:0..nn.layers[0].size-1 {
                instance.layer_states[0].activation[p] = training_data[i].pixels[p] / 255.0;
            }
            propagate_forward(nn, instance);

            if cfg.loss_func == .MSE {
                memset(ys.data, 0, ys.count * size_of(float));
                ys[training_data[i].value] = 1.0;
                output_layer := nn.layers.count - 1;
                for j:0..nn.layers[output_layer].size-1 {
                    loss += square(instance.layer_states[output_layer].activation[j] - ys[j]);
                }
            }
            else {
                output_layer := nn.layers.count - 1;
                softmax(instance.layer_states[output_layer].activation, ys);
                prob := ys[training_data[i].value];
                loss -= log(prob + 0.00000001);
            }
        }
        loss /= cast(float) training_data.count;
        normalized_loss = ifx cfg.loss_func == .MSE then loss else (loss / log(cast(float) ys.count));
    
        print ("Finished epoch % in %ms, normalized loss: %\n", epoch, 
            to_milliseconds(current_time_consensus() - start), normalized_loss);
    }

    array_free(ys);
    array_free(indices);
    print ("Finished training\n");
    nn.loss = normalized_loss;
    return normalized_loss;
}

load_data :: (data_set: DataSet, training_data: *[..]TestData, test_data: *[..]TestData)
{
    train_data_source : string = ---;
    train_data_bin : string = ---;
    test_data_source : string = ---;
    test_data_bin : string = ---;
    if data_set == .EMNIST_Balanced {
        train_data_source = "data/emnist_balanced_train.csv";
        train_data_bin    = "data/emnist_balanced_train.bin";
        test_data_source  = "data/emnist_balanced_test.csv";
        test_data_bin     = "data/emnist_balanced_test.bin";
    }
    else {
        train_data_source = "data/mnist_train.csv";
        train_data_bin    = "data/mnist_train.bin";
        test_data_source  = "data/mnist_test.csv";
        test_data_bin     = "data/mnist_test.bin";
    }
    file:, success: = file_open(train_data_bin);
    if (!success) {
        convert_csv_to_bin(train_data_source, train_data_bin);
    }
    file, success = file_open(test_data_bin);
    if (!success) {
        convert_csv_to_bin(test_data_source, test_data_bin);
    }
    
    success = load_data_bin(train_data_bin, training_data);
    assert(success);
    print ("Loaded training data\n");
    success = load_data_bin(test_data_bin, test_data);
    assert(success);
    print ("Loaded test data\n");
}

convert_csv_to_bin ::  (csv_file: string, bin_file: string) -> bool
{
    data: [..]TestData;
    defer array_free(data);
    success := load_data_csv(csv_file, *data);
    if !success {
        return false;
    }
    result : [..]u8;
    defer array_free(result);
    // pixel count + 1 for value
    entry_size: u32 = data[0].pixels.count + 1;
    array_resize(*result, 4 + data.count * entry_size);
    memcpy(result.data, *entry_size, 4);
    for curr, i : data {
        offset := 4 + (entry_size * i);
        memcpy(result.data + offset, *curr.value, 1);
        memcpy(result.data + offset + 1, curr.pixels.data, (entry_size - 1));
    }
    return write_entire_file(bin_file, result.data, result.count);
}

load_data_bin :: (bin_file: string, result: *[..]TestData) -> bool
{
    bin_data, success := read_entire_file(bin_file);
    if !success {
        log_error("Unable to read file '%'.\n", bin_file);
        return false;
    }
    defer free(bin_data);
    
    entry_size : u32;
    memcpy(*entry_size, bin_data.data, 4);
    num_records := bin_data.count / entry_size;
    array_resize(result, num_records);
    print("Loaded binary data with % entries\n", num_records);
    for i:0..num_records-1 {
        result.data[i].value = bin_data[4 + (i * entry_size)];
        memcpy(result.data[i].pixels.data, bin_data.data + 4 + 1 + (i * entry_size), (entry_size - 1));
    }
    return true;
}

load_data_csv :: (csv_file: string, result: *[..]TestData) -> bool
{
    csv_data, success := read_entire_file(csv_file);
    if !success {
        log_error("Unable to read file '%'.\n", csv_file);
        return false;
    }
    defer free(csv_data);

    lines := split(csv_data, "\n");
    entry : TestData;
    for line, i : lines {
        if (line.count == 0) {
            // skip blank lines (normally at EOF)
            continue;
        }
        values := split(line, ",");
        entry.value = string_to_int(values[0], 10, u8);
        for p:1..values.count-1 {
            entry.pixels[p - 1] = string_to_int(values[p], 10, u8);
        }
        array_add(result, entry);
    }
    return true;
}

configure_layers :: (training_config: *TrainingConfig)
{
    input_size : u32 = TrainingConfig.IMAGE_SIZE * TrainingConfig.IMAGE_SIZE;
    array_resize(*training_config.layers, 2 + training_config.hidden_layers.count);
    // Note: the activation is N/A for the input layer, it could have its own
    // representation without the activation function but we just ignore it
    training_config.layers[0] = .{  input_size, .Linear };
    for i:0..training_config.hidden_layers.count-1 {
        training_config.layers[i + 1] = training_config.hidden_layers[i];
    }
    output_layer_size := ifx (training_config.data_set == .EMNIST_Balanced) 
        then EMNIST_BALANCED_MAPPING.count else MNIST_MAPPING.count;
    training_config.layers[1 + training_config.hidden_layers.count] = .{ xx output_layer_size, .Linear };
}

configure_layers_default :: (training_config: *TrainingConfig)
{
    array_resize(*training_config.hidden_layers, 2);
    training_config.hidden_layers[0].size = 36;
    training_config.hidden_layers[0].activation = .ReLU;
    training_config.hidden_layers[1].size = 25;
    training_config.hidden_layers[1].activation = .ReLU;
    configure_layers(training_config);
}
