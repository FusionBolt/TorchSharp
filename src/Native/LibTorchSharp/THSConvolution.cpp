// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSNN.h"

#include <torch/nn/init.h>



NNModule THSNN_AvgPool1d_ctor(const int64_t* kernelSize, const int64_t* stride,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AvgPool1dOptions(at::ArrayRef<int64_t>(kernelSize, 1));
    if (stride)
        opts = opts.stride(at::ArrayRef<int64_t>(stride, 1));
    res = create_module<torch::nn::AvgPool1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AvgPool1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AvgPool1d>()->forward(*tensor));
}

NNModule THSNN_AvgPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AvgPool2dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
    if (stride)
        opts = opts.stride(at::ArrayRef<int64_t>(stride, strideLength));
    res = create_module<torch::nn::AvgPool2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AvgPool2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AvgPool2d>()->forward(*tensor));
}

NNModule THSNN_AvgPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AvgPool3dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
    if (stride)
        opts = opts.stride(at::ArrayRef<int64_t>(stride, strideLength));
    res = create_module<torch::nn::AvgPool3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AvgPool3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AvgPool3d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveAvgPool1d_ctor(const int64_t* kernelSize, const int kernelSizeLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AdaptiveAvgPool1dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
    res = create_module<torch::nn::AdaptiveAvgPool1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AdaptiveAvgPool1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AdaptiveAvgPool1d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveAvgPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AdaptiveAvgPool2dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
    res = create_module<torch::nn::AdaptiveAvgPool2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AdaptiveAvgPool2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AdaptiveAvgPool2d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveAvgPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AdaptiveAvgPool3dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
    res = create_module<torch::nn::AdaptiveAvgPool3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AdaptiveAvgPool3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AdaptiveAvgPool3d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveMaxPool1d_ctor(const int64_t* kernelSize, const int kernelSizeLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AdaptiveMaxPool1dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
    res = create_module<torch::nn::AdaptiveMaxPool1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AdaptiveMaxPool1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AdaptiveMaxPool1d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveMaxPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AdaptiveMaxPool2dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
    res = create_module<torch::nn::AdaptiveMaxPool2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AdaptiveMaxPool2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AdaptiveMaxPool2d>()->forward(*tensor));
}

NNModule THSNN_AdaptiveMaxPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::AdaptiveMaxPool3dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength));
    res = create_module<torch::nn::AdaptiveMaxPool3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_AdaptiveMaxPool3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::AdaptiveMaxPool3d>()->forward(*tensor));
}

NNModule THSNN_MaxPool1d_ctor(const int64_t* kernelSize, const int64_t* stride, const int64_t* padding, const int64_t* dilation, bool ceil_mode,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::MaxPool1dOptions(at::ArrayRef<int64_t>(kernelSize, 1)).ceil_mode(ceil_mode);
        if (stride)
            opts = opts.stride(at::ArrayRef<int64_t>(stride, 1));
        if (padding)
            opts = opts.padding(at::ArrayRef<int64_t>(padding, 1));
        if (dilation)
            opts = opts.dilation(at::ArrayRef<int64_t>(dilation, 1));

        res = create_module<torch::nn::MaxPool1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_MaxPool1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::MaxPool1d>()->forward(*tensor));
}

Tensor THSNN_MaxPool1d_forward_with_indices(const NNModule module, const Tensor tensor, Tensor* indices)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = (*module)->as<torch::nn::MaxPool1d>()->forward_with_indices(*tensor););
    *indices = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

NNModule THSNN_MaxPool2d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength, const int64_t* dilation, const int dilationLength, bool ceil_mode,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::MaxPool2dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength)).ceil_mode(ceil_mode);
        if (stride)
            opts = opts.stride(at::ArrayRef<int64_t>(stride, strideLength));
        if (padding)
            opts = opts.padding(at::ArrayRef<int64_t>(padding, paddingLength));
        if (dilation)
            opts = opts.dilation(at::ArrayRef<int64_t>(dilation, dilationLength));

        res = create_module<torch::nn::MaxPool2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_MaxPool2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::MaxPool2d>()->forward(*tensor));
}

Tensor THSNN_MaxPool2d_forward_with_indices(const NNModule module, const Tensor tensor, Tensor* indices)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = (*module)->as<torch::nn::MaxPool2d>()->forward_with_indices(*tensor););
    *indices = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

NNModule THSNN_MaxPool3d_ctor(const int64_t* kernelSize, const int kernelSizeLength, const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength, const int64_t* dilation, const int dilationLength, bool ceil_mode,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::MaxPool3dOptions(at::ArrayRef<int64_t>(kernelSize, kernelSizeLength)).ceil_mode(ceil_mode);
        if (stride)
            opts = opts.stride(at::ArrayRef<int64_t>(stride, strideLength));
        if (padding)
            opts = opts.padding(at::ArrayRef<int64_t>(padding, paddingLength));
        if (dilation)
            opts = opts.dilation(at::ArrayRef<int64_t>(dilation, dilationLength));

        res = create_module<torch::nn::MaxPool3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_MaxPool3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::MaxPool3d>()->forward(*tensor));
}

Tensor THSNN_MaxPool3d_forward_with_indices(const NNModule module, const Tensor tensor, Tensor* indices)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = (*module)->as<torch::nn::MaxPool3d>()->forward_with_indices(*tensor););
    *indices = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}


NNModule THSNN_ZeroPad2d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ZeroPad2dOptions(padding);
    res = create_module<torch::nn::ZeroPad2dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ZeroPad2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ZeroPad2d>()->forward(*tensor));
}

NNModule THSNN_ConstantPad1d_ctor(const double value, const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConstantPad1dOptions(padding, value);
    res = create_module<torch::nn::ConstantPad1dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ConstantPad1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConstantPad1d>()->forward(*tensor));
}

NNModule THSNN_ConstantPad2d_ctor(const double value, const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConstantPad2dOptions(padding, value);
    res = create_module<torch::nn::ConstantPad2dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ConstantPad2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConstantPad2d>()->forward(*tensor));
}

NNModule THSNN_ConstantPad3d_ctor(const double value, const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConstantPad3dOptions(padding, value);
    res = create_module<torch::nn::ConstantPad3dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ConstantPad3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConstantPad3d>()->forward(*tensor));
}

NNModule THSNN_ReplicationPad1d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReplicationPad1dOptions(padding);
    res = create_module<torch::nn::ReplicationPad1dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ReplicationPad1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReplicationPad1d>()->forward(*tensor));
}

NNModule THSNN_ReplicationPad2d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReplicationPad2dOptions(padding);
    res = create_module<torch::nn::ReplicationPad2dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ReplicationPad2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReplicationPad2d>()->forward(*tensor));
}

NNModule THSNN_ReplicationPad3d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReplicationPad3dOptions(padding);
    res = create_module<torch::nn::ReplicationPad3dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ReplicationPad3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReplicationPad3d>()->forward(*tensor));
}

NNModule THSNN_ReflectionPad1d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReflectionPad1dOptions(padding);
    res = create_module<torch::nn::ReflectionPad1dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ReflectionPad1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReflectionPad1d>()->forward(*tensor));
}

NNModule THSNN_ReflectionPad2d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReflectionPad2dOptions(padding);
    res = create_module<torch::nn::ReflectionPad2dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ReflectionPad2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReflectionPad2d>()->forward(*tensor));
}

NNModule THSNN_ReflectionPad3d_ctor(const int64_t padding, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReflectionPad3dOptions(padding);
    res = create_module<torch::nn::ReflectionPad3dImpl>(opts, outAsAnyModule);
    );
}

Tensor   THSNN_ReflectionPad3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReflectionPad3d>()->forward(*tensor));
}



template<typename T>
void ApplyPaddingMode(T& opts, const int64_t padding)
{
    if (padding == 0)
        opts = opts.padding_mode(torch::kZeros);
    if (padding == 1)
        opts = opts.padding_mode(torch::kReflect);
    if (padding == 2)
        opts = opts.padding_mode(torch::kReplicate);
    if (padding == 3)
        opts = opts.padding_mode(torch::kCircular);
}

NNModule THSNN_Conv1d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    torch::nn::Conv1dOptions::padding_t padd(padding);
    if (padding == -1)
    {
        padd = torch::kSame;
    }

    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv1dOptions(inputChannel, outputChannel, kernelSize)
            .stride(stride)
            .padding(padd)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);

        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Conv1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Conv1d>()->forward(*tensor));
}

Tensor THSNN_Conv1d_bias(const NNModule module)
{
    return get_bias<torch::nn::Conv1d>(module);
}

void THSNN_Conv1d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::Conv1d>(module, bias);
}

Tensor THSNN_Conv1d_weight(const NNModule module)
{
    return get_weight<torch::nn::Conv1d>(module);
}

void THSNN_Conv1d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::Conv1d>(module, weight);
}

NNModule THSNN_Conv2d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    torch::nn::Conv2dOptions::padding_t padd =
        (padding == -1) ? torch::kSame :
        (padding == 0) ? torch::kValid :
        torch::nn::Conv2dOptions::padding_t(padding);

    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv2dOptions(inputChannel, outputChannel, kernelSize)
            .stride(stride)
            .padding(padd)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv2dImpl>(opts, outAsAnyModule);
    );
}

NNModule THSNN_Conv2d_ctor_1(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelX, const int64_t kernelY,
    const int64_t strideX, const int64_t strideY,
    const int64_t paddingX, const int64_t paddingY,
    const int64_t dilationX, const int64_t dilationY,
    const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    torch::nn::Conv2dOptions::padding_t padd =
        (paddingX == -1) ? torch::kSame :
        (paddingX == 0) ? torch::kValid :
        (torch::nn::Conv2dOptions::padding_t)torch::ExpandingArray<2>({ paddingX, paddingY });

    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv2dOptions(inputChannel, outputChannel, { kernelX, kernelY })
        .stride({ strideX, strideY })
        .padding(padd)
        .dilation({ dilationX, dilationY })
        .groups(groups)
        .bias(bias);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Conv2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Conv2d>()->forward(*tensor));
}

Tensor THSNN_Conv2d_bias(const NNModule module)
{
    return get_bias<torch::nn::Conv2d>(module);
}

void THSNN_Conv2d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::Conv2d>(module, bias);
}

Tensor THSNN_Conv2d_weight(const NNModule module)
{
    return get_weight<torch::nn::Conv2d>(module);
}

void THSNN_Conv2d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::Conv2d>(module, weight);
}

NNModule THSNN_Conv3d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    torch::nn::Conv3dOptions::padding_t padd =
        (padding == -1) ? torch::kSame :
        (padding == 0) ? torch::kValid :
        torch::nn::Conv3dOptions::padding_t(padding);

    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv3dOptions(inputChannel, outputChannel, kernelSize)
            .stride(stride)
            .padding(padd)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv3dImpl>(opts, outAsAnyModule);
    );
}

NNModule THSNN_Conv3d_ctor_1(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelX, const int64_t kernelY, const int64_t kernelZ,
    const int64_t strideX, const int64_t strideY, const int64_t strideZ,
    const int64_t paddingX, const int64_t paddingY, const int64_t paddingZ,
    const int64_t dilationX, const int64_t dilationY, const int64_t dilationZ,
    const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    torch::nn::Conv3dOptions::padding_t padd =
        (paddingX == -1) ? torch::kSame :
        (paddingX == 0) ? torch::kValid :
        (torch::nn::Conv3dOptions::padding_t)torch::ExpandingArray<3>({ paddingX, paddingY, paddingZ });

    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv3dOptions(inputChannel, outputChannel, { kernelX, kernelY, kernelZ })
        .stride({ strideX, strideY, strideZ })
        .padding(padd)
        .dilation({ dilationX, dilationY, dilationZ })
        .groups(groups)
        .bias(bias);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Conv3d_bias(const NNModule module)
{
    return get_bias<torch::nn::Conv3d>(module);
}

void THSNN_Conv3d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::Conv3d>(module, bias);
}

Tensor THSNN_Conv3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Conv3d>()->forward(*tensor));
}

Tensor THSNN_Conv3d_weight(const NNModule module)
{
    return get_weight<torch::nn::Conv3d>(module);
}

void THSNN_Conv3d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::Conv3d>(module, weight);
}


NNModule THSNN_ConvTranspose1d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t output_padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConvTranspose1dOptions(inputChannel, outputChannel, kernelSize)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias)
        .output_padding(output_padding);
    ApplyPaddingMode(opts, paddingMode);

    res = create_module<torch::nn::ConvTranspose1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ConvTranspose1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConvTranspose1d>()->forward(*tensor));
}

Tensor THSNN_ConvTranspose1d_bias(const NNModule module)
{
    return get_bias<torch::nn::ConvTranspose1d>(module);
}

void THSNN_ConvTranspose1d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::ConvTranspose1d>(module, bias);
}

Tensor THSNN_ConvTranspose1d_weight(const NNModule module)
{
    return get_weight<torch::nn::ConvTranspose1d>(module);
}

void THSNN_ConvTranspose1d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::ConvTranspose1d>(module, weight);
}

NNModule THSNN_ConvTranspose2d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t output_padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConvTranspose2dOptions(inputChannel, outputChannel, kernelSize)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias)
        .output_padding(output_padding);
    ApplyPaddingMode(opts, paddingMode);

    res = create_module<torch::nn::ConvTranspose2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ConvTranspose2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConvTranspose2d>()->forward(*tensor));
}

Tensor THSNN_ConvTranspose2d_bias(const NNModule module)
{
    return get_bias<torch::nn::ConvTranspose2d>(module);
}

void THSNN_ConvTranspose2d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::ConvTranspose2d>(module, bias);
}

Tensor THSNN_ConvTranspose2d_weight(const NNModule module)
{
    return get_weight<torch::nn::ConvTranspose2d>(module);
}

void THSNN_ConvTranspose2d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::ConvTranspose2d>(module, weight);
}

NNModule THSNN_ConvTranspose3d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t output_padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConvTranspose3dOptions(inputChannel, outputChannel, kernelSize)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias)
        .output_padding(output_padding);
    ApplyPaddingMode(opts, paddingMode);

    res = create_module<torch::nn::ConvTranspose3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ConvTranspose3d_bias(const NNModule module)
{
    return get_bias<torch::nn::ConvTranspose3d>(module);
}

void THSNN_ConvTranspose3d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::ConvTranspose3d>(module, bias);
}

Tensor THSNN_ConvTranspose3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConvTranspose3d>()->forward(*tensor));
}

Tensor THSNN_ConvTranspose3d_weight(const NNModule module)
{
    return get_weight<torch::nn::ConvTranspose3d>(module);
}

void THSNN_ConvTranspose3d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::ConvTranspose3d>(module, weight);
}
