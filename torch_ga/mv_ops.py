"""Operations on geometric algebra tensors used internally."""
from typing import Union
import torch


def mv_multiply(a_blade_values: torch.Tensor, b_blade_values: torch.Tensor, cayley: torch.Tensor) -> torch.Tensor:
    # x = torch.einsum("i,j,ijk->k", a_blade_values, b_blade_values, cayley)
    

    # cehck later
    # # # ...i, ijk -> ...jk
    # # x = torch.tensordot(a_blade_values, cayley, dims=[-1, 0])
    # x = torch.tensordot(a_blade_values, cayley, dims=([-1, 0],[-1,0]))
    # # # ...1j, ...jk -> ...1k
    # # x = tf.expand_dims(b_blade_values, axis=b_blade_values.shape.ndims - 1) @ x
    # x = b_blade_values.unsqueeze(len(b_blade_values.shape) - 1) @ x
    # # # ...1k -> ...k
    # # x = torch.squeeze(x, axis=-2)
    # x = torch.squeeze(x, axis=-2)
    
    # cehck later
    # # ...i, ijk -> ...jk
    # x = torch.tensordot(a_blade_values, cayley, dims=[-1, 0])
    # x = torch.tensordot(a_blade_values, cayley, dims=([-1, 0],[-1,0]))
    x = torch.tensordot(a_blade_values, cayley, dims=([-1],[0]))

    # # ...1j, ...jk -> ...1k
    # x = tf.expand_dims(b_blade_values, axis=b_blade_values.shape.ndims - 1) @ x
    # x = b_blade_values.unsqueeze(len(b_blade_values.shape) - 1) @ x
    x = b_blade_values.unsqueeze(-2) @ x        
    # # ...1k -> ...k
    # x = torch.squeeze(x, axis=-2)
    x = torch.squeeze(x, axis=-2) 
       
    # # # ...1j, ...jk -> ...1k
    # x = b_blade_values @ x        
    
    # print(f"same opeartions? x.shape={x.shape},x1.shape={x1.shape}")
    
    # # einsum
    # x1 = torch.einsum("...i,...j,ijk->...k", a_blade_values, b_blade_values, cayley)    
    # assert(torch.all(torch.isclose(x1,x))), f"should be the same operation x[0]={x[0]}, x1[0]={x1[0]}"
    

    return x


def mv_multiply_element_wise(a_blade_values: torch.Tensor, b_blade_values: torch.Tensor, cayley: torch.Tensor) -> torch.Tensor:
    x = a_blade_values * b_blade_values
    return x

import math
import torch.nn.functional as F
# https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/10
def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    # x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
    x = F.pad(x, (pad_col//2, pad_col - pad_col//2, pad_row//2, pad_row - pad_row//2, ))
    
    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    
    return patches.view(b,-1,patches.shape[-2], patches.shape[-1])

# https://discuss.pytorch.org/t/conv1d-implementation-using-torch-nn-functional-unfold/109643/3
import torch.nn.functional as F
def f_mv_conv1d(input, weight, cayley: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    input : input tensor of shape    : (minibatch,in_channels, iW)
    input : input tensor of shape    : (minibatch,in_channels, width, num_blades)
    weight : filters of shape : (out_channels, in_channels/groups, kW)
    weight : filters of shape : (out_channels, in_channels/groups, kernel_size, num_blades)
    bias :  optional bias of shape (out_channels). Default: None
    bias :  optional bias of shape (out_channels, num_blades). Default: None
    stride :  the stride of the convolving kernel. Can be a single number or a one-element tuple (sW,). Default: 1
    padding : implicit paddings on both sides of the input. Can be a string {‘valid’, ‘same’}, single number or a one-element tuple (padW,). Default: 0 padding='valid' is the same as no padding. padding='same' pads the input so the output has the same shape as the input. However, this mode doesn’t support any stride values other than 1. 
    dilation : the spacing between kernel elements. Can be a single number or a one-element tuple (dW,). Default: 1
    groups : split input into groups, in_channels should be divisible by the number of groups. Default: 1    
    """
    # kernel_size = weight.shape

    assert len(input.shape)==4, "input size == 4 (minibatch,in_channels, width, num_blades)"
    assert len(weight.shape)==4, "weights size == 4 (out_channels, in_channels/groups, kernel_size, num_blades)"

    # A: [..., S, CI, BI]
    # K: [K, CI, CO, BK]
    # C: [BI, BK, BO]    
    input = input.permute(0,2,1,3)
    weight = weight.permute(2,1,0,3)

    batch,in_channels,width,num_blades = input.shape
    out_channels, in_channels, kernel,num_blades1 = weight.shape
    assert (num_blades==num_blades1), "same geometry please"
    kernel_size = (kernel,num_blades)

    input_unfold = F.unfold(input.view(batch * groups, in_channels // groups, width, num_blades), kernel_size, dilation, 0, stride)
    # N,Cxprod_kernel,L
    # input_unfold = input_unfold.view(batch, groups, input_unfold.size(1), input_unfold.size(2))
    input_unfold = input_unfold.view(batch, in_channels // groups, kernel, num_blades, input_unfold.size(2))
    # ci,ks,bi,L * co,ci,ks,bj * bi,bj,bk -> co,ks,L,bk  
    # a,b,c,d * e,a,b,f * c,f,g -> e,b,d,g
    # ...abcd, eabf, cfg -> ...ebdg  
    x = torch.einsum("...abcd, eabf, cfg -> ...ebdg", input_unfold, weight, cayley)
    x = x.view(batch,out_channels,-1,num_blades) + (bias.view(1,out_channels,1,num_blades) if bias else 0) 
    x = x.permute(0,2,1,3)
    return x

    # input = input.unqueeze(3) #now size is 4
    # input_unfold = F.unfold(input, kernel_size, dilation, padding, stride)
    # out_unfold = input_unfold.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
    # # input, output_size, kernel_size, dilation=1, padding=0, stride=1
    # F.fold(out_unfold, output_size, (1, 1),dilation, padding, stride)
    

import numpy as np
def mv_conv1d(a_blade_values: torch.Tensor, k_blade_values: torch.Tensor, cayley: torch.Tensor,
              stride: int, padding: str, dilations: Union[int, None] = None) -> torch.Tensor:
    # Winograd convolution

    # A: [..., S, CI, BI]
    # K: [K, CI, CO, BK]
    # C: [BI, BK, BO]

    kernel_size = k_blade_values.shape[0]

    a_batch_shape = a_blade_values.shape[:-3]

    # Reshape a_blade_values to a 2d image (since that's what the tf op expects)
    # [*, S, 1, CI*BI]
    # a_image_shape = torch.concat([
    #     torch.tensor(a_batch_shape),
    #     torch.tensor(a_blade_values.shape[-3:-2]),
    #     torch.tensor([1, torch.prod(torch.tensor(a_blade_values.shape[-2:]))])
    # ], axis=0)
    a_image_shape = list(a_batch_shape) + list(a_blade_values.shape[-3:-2]) + [1, np.prod(a_blade_values.shape[-2:]) ]
    print(f"a_image_shape={a_image_shape}")
    a_image = torch.reshape(a_blade_values, tuple([int(_) for _ in a_image_shape]))

    sizes = [1, kernel_size, 1, 1]
    strides = [1, stride, 1, 1]

    # [*, P, 1, K*CI*BI] where eg. number of patches P = S * K for
    # stride=1 and "SAME", (S-K+1) * K for "VALID", ...
    # a_slices = tf.image.extract_patches(
    #     a_image,
    #     sizes=sizes, strides=strides,
    #     rates=[1, 1, 1, 1], padding=padding
    # )
    # extract_image_patches(x, kernel, stride=1, dilation=1):
    a_slices = extract_image_patches(
        a_image,
        sizes, stride=strides
        # rates=[1, 1, 1, 1], 
        # padding=padding
    )

    # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    # inp_unf = F.unfold(a_image, kernel_size=sizes, dilation=1, padding=padding, stride=strides)
    # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
    # out = F.fold(out_unf, (7, 8), (1, 1))

    # [..., P, K, CI, BI]
    out_shape = torch.concat([
        a_batch_shape,
        a_slices.shape()[-3:-2],
        k_blade_values.shape()[:1],
        a_blade_values.shape()[-2:]
    ], axis=0)

    a_slices = torch.reshape(a_slices, out_shape)

    # TODO: Optimize this to not use einsum (since it's slow with ellipses)
    # a_...p,k,ci,bi; k_k,ci,co,bk; c_bi,bk,bo -> y_...p,co,bo
    #   ...a b c  d ,   e c  f  g ,   d  g  h  ->   ...a f  h
    x = torch.einsum("...abcd,bcfg,dgh->...afh", a_slices, k_blade_values, cayley)
    return x


def mv_reversion(a_blade_values, algebra_blade_degrees):
    algebra_blade_degrees = algebra_blade_degrees.to(torch.float32)
    # for each blade, 0 if even number of swaps required, else 1
    odd_swaps =  (torch.floor(algebra_blade_degrees * (algebra_blade_degrees - 0.5)) % 2).to(dtype=torch.float32)
    # [0, 1] -> [-1, 1]
    reversion_signs = 1.0 - 2.0 * odd_swaps
    return reversion_signs * a_blade_values


def mv_grade_automorphism(a_blade_values, algebra_blade_degrees):
    algebra_blade_degrees = algebra_blade_degrees.to(dtype=torch.float32)
    signs = 1.0 - 2.0 * (algebra_blade_degrees % 2.0)
    return signs * a_blade_values
