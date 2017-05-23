//
// Created by hacht on 5/1/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

#include "time.h"

namespace simpleCNN {

#ifdef USE_CUDNN
  TEST(Cuda, print_device_info) {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
  }

#define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag                 = (default_value)
#define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
#define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag                 = (default_value)
#define DEFINE_double(flag, default_value, description) const double FLAGS_##flag             = (default_value)
#define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag((default_value))

  DEFINE_int32(gpu, 0, "The GPU ID to use");

  TEST(Cuda, testing) {
    /*std::vector<float_t> x = {1, 2, 3};

    float_t* x_gpu = cuda_make_array(&x[0], x.size());

    float_t x_cpu[3];

    cuda_pull_array(x_gpu, x_cpu, x.size());

    for (size_t i = 0; i < 3; ++i) {
      ASSERT_EQ(x_cpu[i], x[i]);
    }

    cuda_free(x_gpu);*/
  }

  TEST(Cuda, copy_back_and_forth_tensor) {
    tensor_t test({2, 2, 2, 2});
    uniform_rand(test.host_begin(), test.host_end(), 0, 1);

    float_t* x_gpu = cuda_make_array(test.host_ptr(0, 0, 0, 0), test.size());

    float_t x_cpu[test.size()];
    cuda_pull_array(x_gpu, x_cpu, test.size());

    for (size_t i = 0; i < test.size(); ++i) {
      ASSERT_EQ(x_cpu[i], *(test.host_begin() + i));
    }

    cuda_free(x_gpu);
  }

  TEST(Cuda, cudnnConvolution) {
    size_t padding = 1;
    size_t stride  = 2;
    size_t w       = 5;
    size_t h       = 5;
    size_t ch      = 3;  // 3 color channel
    size_t bs      = 1;
    size_t fs      = 3;
    size_t o_ch    = 2;  // := number of filters
    size_t o_w     = 3;
    size_t o_h     = 3;

    tensor_t image_cpu({bs, ch, h, w});
    vec_t image_data = {0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1, 1, 2, 2, 1, 0, 2, 0, 1,
                        1, 0, 2, 2, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                        2, 1, 2, 0, 1, 0, 0, 1, 2, 2, 1, 1, 1, 0, 2, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2};
    fill(image_data, image_cpu);
    float_t* image_gpu = cuda_make_array(image_cpu.host_ptr(0, 0, 0, 0), image_cpu.size());

    tensor_t weight_cpu({o_ch, ch, fs, fs});
    vec_t weight_data = {-1, -1, 0,  1, -1, 0,  -1, -1, 0, -1, -1, -1, 0, -1, -1, 1,  0,  -1,
                         -1, 1,  0,  1, 1,  -1, 1,  -1, 0, 1,  0,  -1, 1, -1, 0,  -1, 1,  -1,
                         1,  1,  -1, 0, 1,  -1, 1,  0,  1, 0,  0,  1,  0, 1,  1,  1,  -1, 1};
    fill(weight_data, weight_cpu);
    float_t* weight_gpu = cuda_make_array(weight_cpu.host_ptr(0, 0, 0, 0), weight_cpu.size());

    tensor_t output_cpu({bs, o_ch, o_h, o_w});
    output_cpu.fill(0.0f);
    float_t* output_gpu = cuda_make_array(output_cpu.host_ptr(0, 0, 0, 0), output_cpu.size());

    cudnnHandle_t cudnnHandle;
    checkCUDNN(cudnnCreate(&cudnnHandle));

    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;

    checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&weightDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bs, ch, h, w));
    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bs, o_ch, o_h, o_w));
    checkCUDNN(cudnnSetFilter4dDescriptor(weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, o_ch, ch, fs, fs));
    checkCUDNN(
      cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION));
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, srcTensorDesc, weightDesc, convDesc, dstTensorDesc,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fw_algo));

    size_t sizeInBytes;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, srcTensorDesc, weightDesc, convDesc, dstTensorDesc,
                                                       fw_algo, &sizeInBytes));
    ;

    void* workspace = nullptr;
    if (sizeInBytes > 0) {
      checkCudaErrors(cudaMalloc(&workspace, sizeInBytes));
    }

    float_t one = 1;
    cudnnConvolutionForward(cudnnHandle, &one, srcTensorDesc, image_gpu, weightDesc, weight_gpu, convDesc, fw_algo,
                            workspace, sizeInBytes, &one, dstTensorDesc, output_gpu);

    cuda_pull_array(output_gpu, output_cpu.host_ptr(0, 0, 0, 0), output_cpu.size());

    vec_t correct_output = {-1, -5, 0, -4, -4, -7, -1, -5, -4, 4, 1, 2, -1, 3, 9, 1, 3, 4};

    auto iter = output_cpu.host_begin();
    for (auto d : correct_output) {
      ASSERT_EQ(d, *iter++);
    }

    cuda_free(image_gpu);
    cuda_free(weight_gpu);
    cuda_free(output_gpu);

    checkCudaErrors(cudaFree(workspace));
  }

  TEST(Cuda, forward_prop) {
    size_t padding = 1;
    size_t stride  = 2;
    size_t w       = 5;
    size_t h       = 5;
    size_t ch      = 3;  // 3 color channel
    size_t bs      = 1;
    size_t fs      = 3;
    size_t o_ch    = 2;  // := number of filters
    size_t o_w     = 3;
    size_t o_h     = 3;

    using conv = Convolutional_layer;

    conv c(w, h, ch, bs, fs, o_ch, stride, padding, true, core::backend_t::gpu);

    tensor_t in_cpu({bs, ch, h, w});
    vec_t in_data = {0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1, 1, 2, 2, 1, 0, 2, 0, 1,
                     1, 0, 2, 2, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                     2, 1, 2, 0, 1, 0, 0, 1, 2, 2, 1, 1, 1, 0, 2, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2};
    fill(in_data, in_cpu);

    vec_t weight_data = {-1, -1, 0,  1, -1, 0,  -1, -1, 0, -1, -1, -1, 0, -1, -1, 1,  0,  -1,
                         -1, 1,  0,  1, 1,  -1, 1,  -1, 0, 1,  0,  -1, 1, -1, 0,  -1, 1,  -1,
                         1,  1,  -1, 0, 1,  -1, 1,  0,  1, 0,  0,  1,  0, 1,  1,  1,  -1, 1};
    weight_init::Test why(weight_data);
    c.weight_init(why);

    vec_t bias_data = {1, 0};
    weight_init::Test bias(bias_data);
    c.bias_init(bias);

    c.init_weight();

    tensor_t out_cpu({bs, o_ch, o_h, o_w});
    c.set_in_data(in_cpu, component_t::IN_DATA);
    c.set_out_data(out_cpu, component_t::OUT_DATA);

    c.forward();

    vec_t correct_output = {0, -4, 1, -3, -3, -6, 0, -4, -3, 4, 1, 2, -1, 3, 9, 1, 3, 4};

    auto iter = out_cpu.host_begin();
    for (auto d : correct_output) {
      ASSERT_EQ(d, *iter++);
    }
  }

  TEST(Cuda, forward_prop_II) {
    size_t w       = 28;
    size_t h       = 28;
    size_t ich     = 1;
    size_t bs      = 1;
    size_t och     = 32;
    size_t fs      = 5;
    size_t stride  = 1;
    size_t padding = 2;

    using conv = Convolutional_layer;
    conv c(w, h, ich, bs, fs, och, stride, padding, true, core::backend_t::gpu);

    tensor_t in({bs, ich, h, w});
    tensor_t out({bs, och, h, w});
    uniform_rand(in.host_begin(), in.host_end(), -1.0f, 1.0f);

    c.setup(true);

    c.set_in_data(in, component_t::IN_DATA);
    c.set_out_data(out, component_t::OUT_DATA);

    c.forward();
  }

  TEST(Cuda, backward_prop) {
    size_t padding = 1;
    size_t stride  = 2;
    size_t w       = 5;
    size_t h       = 5;
    size_t ow      = 3;
    size_t oh      = 3;
    size_t ich     = 3;  // 3 color channel
    size_t bs      = 2;
    size_t fs      = 3;
    size_t och     = 2;  // := number of filters
    bool has_bias  = true;

    using conv = Convolutional_layer;
    conv c(w, h, ich, bs, fs, och, stride, padding, has_bias, core::backend_t::gpu);

    tensor_t in_cpu({bs, ich, h, w});
    vec_t input_data = {0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1, 1, 2, 2, 1, 0, 2, 0, 1, 1, 0, 2, 2, 1,
                        0, 0, 1, 1, 2, 0, 0, 0, 1, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 2, 0, 1, 0, 0, 1, 2, 2,
                        1, 1, 1, 0, 2, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 0, 1,
                        1, 1, 1, 2, 2, 1, 0, 2, 0, 1, 1, 0, 2, 2, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 2, 2, 0, 1, 1, 0,
                        0, 1, 1, 0, 0, 2, 1, 2, 0, 1, 0, 0, 1, 2, 2, 1, 1, 1, 0, 2, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2};
    fill(input_data, in_cpu);

    tensor_t curr_delta({bs, och, oh, ow});
    vec_t curr_delta_data = {0, -4, 1, -3, -3, -6, 0, -4, -3, 4, 1, 2, -1, 3, 9, 1, 3, 4,
                             0, -4, 1, -3, -3, -6, 0, -4, -3, 4, 1, 2, -1, 3, 9, 1, 3, 4};
    fill(curr_delta_data, curr_delta);

    // Setup custom weights and bias
    vec_t weight_data = {-1, -1, 0,  1, -1, 0,  -1, -1, 0, -1, -1, -1, 0, -1, -1, 1,  0,  -1,
                         -1, 1,  0,  1, 1,  -1, 1,  -1, 0, 1,  0,  -1, 1, -1, 0,  -1, 1,  -1,
                         1,  1,  -1, 0, 1,  -1, 1,  0,  1, 0,  0,  1,  0, 1,  1,  1,  -1, 1};
    weight_init::Test wei(weight_data);

    vec_t bias_data = {1, 0};
    weight_init::Test bias(bias_data);

    // Weight allocation (necessary for custom weights)
    c.weight_init(wei);
    c.bias_init(bias);

    // Fill with values (depends on weight::init class)
    c.init_weight();

    tensor_t prev_delta({bs, ich, h, w});
    tensor_t dW({och, ich, fs, fs});
    tensor_t dB({1, och, 1, 1});

    data_ptrs_t input     = {&in_cpu, c.in_component_data(component_t::WEIGHT), c.in_component_data(component_t::BIAS)};
    data_ptrs_t output    = {};
    data_ptrs_t in_grads  = {&prev_delta, &dW, &dB};
    data_ptrs_t out_grads = {&curr_delta, &curr_delta};

    c.back_propagation(input, output, in_grads, out_grads);

    // print(dW, "dW");
    // print(dB, "dB");
  }

  TEST(Cuda, max_forwardprop) {
    size_t w   = 4;
    size_t h   = 4;
    size_t ich = 2;
    size_t bs  = 2;

    using maxpool = Maxpooling_layer;

    maxpool m(w, h, ich, bs, 2, 2, core::backend_t::gpu);

    vec_t in_data = {1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4, 1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4,
                     1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4, 1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4};
    tensor_t img({bs, ich, h, w});
    fill(in_data, img);

    tensor_t out({bs, ich, 2, 2});

    data_ptrs_t input  = {&img};
    data_ptrs_t output = {&out};
    m.forward_propagation(input, output);

    auto outIter        = out.host_begin();
    vec_t correctOutput = {6, 8, 3, 4, 6, 8, 3, 4, 6, 8, 3, 4, 6, 8, 3, 4};
    for (const auto& d : correctOutput) {
      ASSERT_EQ(*outIter++, d);
    }
  }

  TEST(Cuda, max_backprop) {
    size_t w   = 4;
    size_t h   = 4;
    size_t ich = 1;
    size_t och = 1;
    size_t bs  = 1;

    using maxpool = Maxpooling_layer;

    maxpool m(w, h, ich, bs, 2, 2, core::backend_t::gpu);

    // In order to get max index tensor we have to perform activate forward pass first.
    vec_t in_data = {1, 1, 2, 4, 5, -1, 7, -1, 3, 5, 1, 0, 1, 2, 3, 4};
    tensor_t img({bs, ich, h, w});
    fill(in_data, img);

    tensor_t out({bs, ich, 2, 2});

    data_ptrs_t input  = {&img};
    data_ptrs_t output = {&out};
    m.forward_propagation(input, output);

    tensor_t curr_delta({bs, och, 2, 2});
    vec_t curr_delta_data = {1, 2, 3, 4};
    fill(curr_delta_data, curr_delta);

    tensor_t prev_delta({bs, ich, w, w});
    data_ptrs_t input_grad  = {&prev_delta};
    data_ptrs_t output_grad = {&curr_delta};
    m.back_propagation(input, output, input_grad, output_grad);

    ASSERT_EQ(prev_delta.host_at(0, 0, 1, 0), curr_delta_data[0]);
    ASSERT_EQ(prev_delta.host_at(0, 0, 1, 2), curr_delta_data[1]);
    ASSERT_EQ(prev_delta.host_at(0, 0, 2, 1), curr_delta_data[2]);
    ASSERT_EQ(prev_delta.host_at(0, 0, 3, 3), curr_delta_data[3]);
  }

  TEST(Cuda, forward_prop_connected) {
    size_t in_dim     = 4;
    size_t out_dim    = 2;
    size_t batch_size = 1;

    using fully = Connected_layer;

    fully f(in_dim, out_dim, batch_size, true, core::backend_t::gpu);

    vec_t weight_data = {1, 1, 1, 1, -1, -1, -1, -1};
    vec_t bias_data   = {1, -1};

    weight_init::Test weight(weight_data);
    weight_init::Test bias(bias_data);

    f.weight_init(weight);
    f.bias_init(bias);
    f.init_weight();

    tensor_t in({batch_size, 1, in_dim, 1});
    vec_t in_data = {1, 2, -2, -1};
    fill(in_data, in);

    tensor_t out({batch_size, 1, out_dim, 1});

    data_ptrs_t input  = {&in, f.in_component_data(component_t::WEIGHT), f.in_component_data(component_t::BIAS)};
    data_ptrs_t output = {
      &out,
    };

    f.forward_propagation(input, output);

    auto iter = out.host_begin();
    for (const auto& d : bias_data) {
      ASSERT_EQ(*iter++, d);
    }
  }

  TEST(Cuda, backward_prop_connected) {
    size_t in_dim     = 4;
    size_t out_dim    = 2;
    size_t batch_size = 1;

    using fully = Connected_layer;

    fully f(in_dim, out_dim, batch_size);

    // Input
    tensor_t in({batch_size, 1, in_dim, 1});
    vec_t in_data = {1, 2, -2, -1};
    fill(in_data, in);

    // Weight, bias
    vec_t weight_data = {1, 1, 1, 1, -1, -1, -1, -1};
    vec_t bias_data   = {1, -1};

    weight_init::Test weight(weight_data);
    weight_init::Test bias(bias_data);

    f.weight_init(weight);
    f.bias_init(bias);
    f.init_weight();

    tensor_t dW({batch_size, 1, in_dim, out_dim});
    tensor_t db({batch_size, 1, out_dim, 1});

    // curr grad
    tensor_t curr_grad({batch_size, 1, out_dim, 1});
    vec_t curr_grad_data = {1, -1};
    fill(curr_grad_data, curr_grad);

    // prev grad
    tensor_t prev_grad({batch_size, 1, in_dim, 1});

    data_ptrs_t input    = {&in, f.in_component_data(component_t::WEIGHT), f.in_component_data(component_t::BIAS)};
    data_ptrs_t output   = {};
    data_ptrs_t in_grad  = {&prev_grad, &dW, &db};
    data_ptrs_t out_grad = {&curr_grad};

    f.back_propagation(input, output, in_grad, out_grad);

    //print(prev_grad, "Prev delta");
    //print(dW, "dW");
    //print(db, "db");

    // print(*in_grad[0], "Previous gradients");
    // print(*in_grad[1], "dW");
    // print(*in_grad[2], "db");

    // print(dW, "dW");
    // print(db, "dB");

    vec_t cdw = {1, 2, -2, -1, -1, -2, 2, 1};
    auto iter = dW.host_begin();
    for (const auto& w : cdw) {
      ASSERT_EQ(*iter++, w);
    }

    vec_t cdb  = {1, -1};
    auto biter = db.host_begin();
    for (const auto& b : cdb) {
         ASSERT_EQ(*biter++, b);
    }
  }

  TEST(Cuda, relu_test) {
    size_t size = 6;
    Activation_layer r({1, 1, size, 1}, core::activation_t::relu, core::backend_t::gpu);

    tensor_t forward({1, 1, size, 1});
    tensor_t backward({1, 1, size, 1});
    uniform_rand(forward.host_begin(), forward.host_end(), -1, 1);

    r.set_in_data(forward, component_t::IN_DATA);
    r.set_out_data(backward, component_t::OUT_DATA);

    r.forward();

    tensor_t curr_delta(backward.shape_v());
    curr_delta.fill(1.0f);
    tensor_t prev_delta(backward.shape_v());
    r.set_out_grad(curr_delta, component_t::OUT_GRAD);
    r.set_in_grad(prev_delta, component_t::IN_GRAD);

    r.backward();

    //print(backward);
    //print(prev_delta);

    // vec_t corr = {1, 0, 0, 0, 0, 0, 0};
    // for (size_t i = 0; i < forward_a.size(); ++i) {
    //    ASSERT_EQ(forward_a.host_at_index(i), corr[i]);
    //  ASSERT_EQ(backward_a.host_at_index(i), corr[i]);
    //}
  }

  TEST(Cuda, network_test) {
    size_t in_dim     = 4;
    size_t out_dim    = 2;
    size_t batch_size = 1;

    Network<Sequential> net;

    using fully = Connected_layer;
    net << fully(in_dim, out_dim, batch_size, true, core::backend_t::gpu);

    tensor_t in({batch_size, 1, in_dim, 1});
    tensor_t out({batch_size, 1, out_dim, 1});

    uniform_rand(in.host_begin(), in.host_end(), -1.0f, 1.0f);

    tensor_t output = net.test(in);
    //print(output, "Output");
  }

  TEST(Cuda, drop_fprop) {
  size_t bs = 2;
  size_t c = 3;

  Dropout_layer dr(0.75, core::backend_t::gpu);
  dr.set_in_shape({bs, 1, c, 1});

  tensor_t input({bs, 1, c, 1});
  tensor_t output({bs, 1, c, 1});

  uniform_rand(input.host_begin(), input.host_end(), -1.0f, 1.0f);
  dr.set_in_data(input, component_t::IN_DATA);
  dr.set_out_data(output, component_t::OUT_DATA);

  dr.forward();

  //print(input, "Input");
  //print(output, "Output");


  dr.set_out_grad(input, component_t::OUT_GRAD);
  dr.set_in_grad(output, component_t::IN_GRAD);

  dr.backward();

  //print(input, "Input");
  //print(output, "Output");

}

#endif
}  // namespace simpleCNN