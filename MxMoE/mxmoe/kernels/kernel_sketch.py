from string import Template

KERNEL_TEMPLATE = Template("""
// template<${TileConfigs}>
__global__ void ${KernelName}_impl (     //
    half** ptr_As,                       //
    half** ptr_Bs,                       //
    half** ptr_scale_zp_a,               //
    half** ptr_scale_zp_b,               //
    MatricesInfo::TC** ptr_Cs,           //
    MatricesInfo::TC** ptr_Ds,           //
    int64_t* ldas,                       //
    int64_t* ldbs,                       //
    int64_t* ldcs,                       //
    int64_t* ldds,                       //
    dim3* problem_sizes,                 //
    const QParams* qbits_list,           //
    int problem_count,                   //
    ${ProblemOffset}
    int* problem_tiles_prefix_sum        //
) ${ImplBody}

""")

API_TEMPLATE = Template("""
// template<${TileConfigs}>
void ${KernelName}(                      //
    MatricesInfo::TA** ptr_As,           //
    MatricesInfo::TB** ptr_Bs,           //
    half** ptr_scale_zp_a,               //
    half** ptr_scale_zp_b,               //
    MatricesInfo::TC** ptr_Cs,           //
    MatricesInfo::TC** ptr_Ds,           //
    int64_t* ldas,                       //
    int64_t* ldbs,                       //
    int64_t* ldcs,                       //
    int64_t* ldds,                       //
    dim3* problem_sizes,                 //
    dim3* h_problem_sizes,               //
    QParams* qbits_list,                 //
    QParams* h_qbits_list,               //
    int problem_count                    //
) ${APIBody}

/////////////////////////////////////////////////////
""")


KERNEL_DECLARATION_TEMPLATE = Template("""
#pragma once

#include "cta_gemm.cuh"
#include "registry.cuh"

namespace mxmoe {

using namespace tiled_gemm;
using MatricesInfo = RCR_FP16FP16FP16;

${Signatures}

}  // namespace mxmoe
""")


KERNEL_DEFINATION_TEMPLATEE = Template("""
#include "cta_gemm.cuh"
#include "${KernelType}_decl.cuh"
#include "tile_scheduler.cuh"

namespace mxmoe {
using namespace tiled_gemm;
using MatricesInfo=RCR_FP16FP16FP16;

${Definitions}

${Register}
} // namespace mxmoe
""")


HZ_FUSE_LAUNCH_TEMPLATE = Template("""{
  using TA = MatricesInfo::TA;
  using TB = MatricesInfo::TB;
  using TC = MatricesInfo::TC;

  auto problem_tiles_prefix_sum = new int[problem_count];
  auto total_tiles              = 0;
  for (int i = 0; i < problem_count; i++) {
    auto a_bits = h_qbits_list[i].qbits.x;
    auto w_bits = h_qbits_list[i].qbits.y;
    auto sym    = h_qbits_list[i].sym;
    auto gsize  = h_qbits_list[i].gsize;

    int M = h_problem_sizes[i].x;
    int N = h_problem_sizes[i].y;

    ${tile_counts}

    problem_tiles_prefix_sum[i] = total_tiles;
  }
  int* d_problem_tiles_prefix_sum;
  checkCudaErrors(cudaMalloc(&d_problem_tiles_prefix_sum, sizeof(int) * problem_count));
  checkCudaErrors(cudaMemcpy(d_problem_tiles_prefix_sum, problem_tiles_prefix_sum, problem_count * sizeof(int), cudaMemcpyHostToDevice));

  // auto kernel     = ${KernelName}_impl<${TileConfigs}>;
  auto kernel     = ${KernelName}_impl;

  int dev_id;
  int num_sm;
  int max_active_blocks;
  int num_threads=32*${num_warps};
  size_t smem_size=${smem_size};
  checkCudaErrors(cudaGetDevice(&dev_id));
  checkCudaErrors(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, num_threads, smem_size));
  int num_ctas = num_sm;
  // int num_ctas = num_sm * num_ctas_per_sm;
  auto launch_cfg = cudaLaunchConfig_t{
      .gridDim          = dim3(num_ctas),
      .blockDim         = dim3(32, ${num_warps}),
      .dynamicSmemBytes = smem_size,
      .stream           = 0,  // use default stream
      .attrs            = nullptr,
      .numAttrs         = 0,  // no attributes
  };
  checkCudaErrors(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  checkCudaErrors(cudaLaunchKernelEx(          //
      &launch_cfg, kernel,                     //
      ptr_As, ptr_Bs,                          //
      ptr_scale_zp_a,                          //
      ptr_scale_zp_b,                          //
      ptr_Cs, ptr_Ds,                          //
      ldas, ldbs, ldcs, ldds,                  //
      problem_sizes,                           //
      qbits_list,                              //
      problem_count,                           //
      d_problem_tiles_prefix_sum  //
      ));

  delete[] problem_tiles_prefix_sum;
  checkCudaErrors(cudaFree(d_problem_tiles_prefix_sum));
}
""")


SEQUENTIAL_LAUNCH_TEMPLATE = Template("""{
  using KernelType = void (*)(                                     //
      half** ptr_As, half** ptr_Bs,                                //
      half** ptr_scale_zp_a,                                       //
      half** ptr_scale_zp_b,                                       //
      half** ptr_Cs, half** ptr_Ds,                                //
      int64_t* ldas, int64_t* ldbs, int64_t* ldcs, int64_t* ldds,  //
      dim3* device_problem_sizes,                                  //
      const QParams* device_qbits_list,                            //
      int problem_count,                                           //
      int problem_offset,                                          //
      int* problem_tiles_prefix_sum                                //
  );

  using TA = MatricesInfo::TA;
  using TB = MatricesInfo::TB;
  using TC = MatricesInfo::TC;

  auto cur_qtype          = h_qbits_list[0];
  auto cur_problem_count  = 0;
  auto cur_problem_offset = 0;
  auto kernel_idx         = 0;


  int total_tiles = 0;
  std::vector<int> problem_tiles_prefix_sum{};
  std::vector<int*> d_problem_tiles_prefix_sum_vec{};

  int dev_id;
  int num_sm;
  checkCudaErrors(cudaGetDevice(&dev_id));
  checkCudaErrors(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  int num_ctas = num_sm;

  KernelType kernels${KernelArray}
  cudaLaunchConfig_t launch_cfgs${LaunchConfigArray}

  // auto get_prefix_sum = [&](int* d_problem_tiles_prefix_sum, int cur_problem_count) {
  //   checkCudaErrors(cudaMalloc(&d_problem_tiles_prefix_sum, sizeof(int) * cur_problem_count));
  //   checkCudaErrors(cudaMemcpy(d_problem_tiles_prefix_sum, problem_tiles_prefix_sum.data(),
  //                              cur_problem_count * sizeof(int), cudaMemcpyHostToDevice));
  //   d_problem_tiles_prefix_sum_vec.push_back(d_problem_tiles_prefix_sum);
  // };

  for (auto i = 0; i < problem_count; i++) {
    auto qtype = h_qbits_list[i];

    int M       = h_problem_sizes[i].x;
    int N       = h_problem_sizes[i].y;
    auto a_bits = h_qbits_list[i].qbits.x;
    auto w_bits = h_qbits_list[i].qbits.y;
    auto sym    = h_qbits_list[i].sym;
    auto gsize  = h_qbits_list[i].gsize;

    if (!(qtype == cur_qtype)) {
      cur_problem_offset = i - cur_problem_count;

      int* d_problem_tiles_prefix_sum;
      // get_prefix_sum(d_problem_tiles_prefix_sum, cur_problem_count);
      checkCudaErrors(cudaMalloc(&d_problem_tiles_prefix_sum, sizeof(int) * cur_problem_count));
      checkCudaErrors(cudaMemcpy(d_problem_tiles_prefix_sum, problem_tiles_prefix_sum.data(),
                                 cur_problem_count * sizeof(int), cudaMemcpyHostToDevice));
      d_problem_tiles_prefix_sum_vec.push_back(d_problem_tiles_prefix_sum);

      auto kernel = kernels[kernel_idx];
      auto launch_cfg = launch_cfgs[kernel_idx];

      checkCudaErrors(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, launch_cfg.dynamicSmemBytes));
      checkCudaErrors(cudaLaunchKernelEx(  //
          &launch_cfg, kernel,             //
          ptr_As, ptr_Bs,                  //
          ptr_scale_zp_a,                  //
          ptr_scale_zp_b,                  //
          ptr_Cs, ptr_Ds,                  //
          ldas, ldbs, ldcs, ldds,          //
          problem_sizes,                   //
          qbits_list,                      //
          cur_problem_count,               //
          cur_problem_offset,              //
          d_problem_tiles_prefix_sum       //
          ));

      cur_problem_count = 1;
      cur_qtype         = qtype;
      kernel_idx++;
      problem_tiles_prefix_sum.clear();
      total_tiles = 0;
    } else {
      cur_problem_count++;
    }

    ${tile_counts}

    if (i == problem_count - 1) {
      cur_problem_offset = problem_count - cur_problem_count;

      int* d_problem_tiles_prefix_sum;
      checkCudaErrors(cudaMalloc(&d_problem_tiles_prefix_sum, sizeof(int) * cur_problem_count));
      checkCudaErrors(cudaMemcpy(d_problem_tiles_prefix_sum, problem_tiles_prefix_sum.data(),
                                 cur_problem_count * sizeof(int), cudaMemcpyHostToDevice));
      d_problem_tiles_prefix_sum_vec.push_back(d_problem_tiles_prefix_sum);

      auto kernel     = kernels[kernel_idx];
      auto launch_cfg = launch_cfgs[kernel_idx];

      checkCudaErrors(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, launch_cfg.dynamicSmemBytes));
      checkCudaErrors(cudaLaunchKernelEx(  //
          &launch_cfg, kernel,             //
          ptr_As, ptr_Bs,                  //
          ptr_scale_zp_a,                  //
          ptr_scale_zp_b,                  //
          ptr_Cs, ptr_Ds,                  //
          ldas, ldbs, ldcs, ldds,          //
          problem_sizes,                   //
          qbits_list,                      //
          cur_problem_count,               //
          cur_problem_offset,              //
          d_problem_tiles_prefix_sum       //
          ));
    }
  }

  // checkCudaErrors(cudaDeviceSynchronize());
  for (auto p : d_problem_tiles_prefix_sum_vec)
    checkCudaErrors(cudaFree(p));
}
""")


MULTI_STREAM_LAUNCH_TEMPLATE = Template("""{
  using KernelType = void (*)(                                     //
      half** ptr_As, half** ptr_Bs,                                //
      half** ptr_scale_zp_a,                                       //
      half** ptr_scale_zp_b,                                       //
      half** ptr_Cs, half** ptr_Ds,                                //
      int64_t* ldas, int64_t* ldbs, int64_t* ldcs, int64_t* ldds,  //
      dim3* device_problem_sizes,                                  //
      const QParams* device_qbits_list,                            //
      int problem_count,                                           //
      int problem_offset,                                          //
      int* problem_tiles_prefix_sum                                //
  );

  using TA = MatricesInfo::TA;
  using TB = MatricesInfo::TB;
  using TC = MatricesInfo::TC;

  auto cur_qtype          = h_qbits_list[0];
  auto cur_problem_count  = 0;
  auto cur_problem_offset = 0;
  auto kernel_idx         = 0;


  int total_tiles = 0;
  std::vector<int> problem_tiles_prefix_sum{};
  std::vector<int*> d_problem_tiles_prefix_sum_vec{};

  int dev_id;
  int num_sm;
  checkCudaErrors(cudaGetDevice(&dev_id));
  checkCudaErrors(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  int num_ctas = num_sm;

  cudaStream_t streams${StreamArray}
  KernelType kernels${KernelArray}
  cudaLaunchConfig_t launch_cfgs${LaunchConfigArray}

  // auto get_prefix_sum = [&](int* d_problem_tiles_prefix_sum, int cur_problem_count) {
  //   checkCudaErrors(cudaMalloc(&d_problem_tiles_prefix_sum, sizeof(int) * cur_problem_count));
  //   checkCudaErrors(cudaMemcpy(d_problem_tiles_prefix_sum, problem_tiles_prefix_sum.data(),
  //                              cur_problem_count * sizeof(int), cudaMemcpyHostToDevice));
  //   d_problem_tiles_prefix_sum_vec.push_back(d_problem_tiles_prefix_sum);
  // };

  for (auto i = 0; i < problem_count; i++) {
    auto qtype = h_qbits_list[i];

    int M       = h_problem_sizes[i].x;
    int N       = h_problem_sizes[i].y;
    auto a_bits = h_qbits_list[i].qbits.x;
    auto w_bits = h_qbits_list[i].qbits.y;
    auto sym    = h_qbits_list[i].sym;
    auto gsize  = h_qbits_list[i].gsize;

    if (!(qtype == cur_qtype)) {
      cur_problem_offset = i - cur_problem_count;

      int* d_problem_tiles_prefix_sum;
      // get_prefix_sum(d_problem_tiles_prefix_sum, cur_problem_count);
      checkCudaErrors(cudaMalloc(&d_problem_tiles_prefix_sum, sizeof(int) * cur_problem_count));
      checkCudaErrors(cudaMemcpy(d_problem_tiles_prefix_sum, problem_tiles_prefix_sum.data(),
                                 cur_problem_count * sizeof(int), cudaMemcpyHostToDevice));
      d_problem_tiles_prefix_sum_vec.push_back(d_problem_tiles_prefix_sum);

      auto kernel = kernels[kernel_idx];
      auto launch_cfg = launch_cfgs[kernel_idx];

      checkCudaErrors(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, launch_cfg.dynamicSmemBytes));
      checkCudaErrors(cudaLaunchKernelEx(  //
          &launch_cfg, kernel,             //
          ptr_As, ptr_Bs,                  //
          ptr_scale_zp_a,                  //
          ptr_scale_zp_b,                  //
          ptr_Cs, ptr_Ds,                  //
          ldas, ldbs, ldcs, ldds,          //
          problem_sizes,                   //
          qbits_list,                      //
          cur_problem_count,               //
          cur_problem_offset,              //
          d_problem_tiles_prefix_sum       //
          ));

      cur_problem_count = 1;
      cur_qtype         = qtype;
      kernel_idx++;
      problem_tiles_prefix_sum.clear();
      total_tiles = 0;
    } else {
      cur_problem_count++;
    }

    ${tile_counts}

    if (i == problem_count - 1) {
      cur_problem_offset = problem_count - cur_problem_count;

      int* d_problem_tiles_prefix_sum;
      checkCudaErrors(cudaMalloc(&d_problem_tiles_prefix_sum, sizeof(int) * cur_problem_count));
      checkCudaErrors(cudaMemcpy(d_problem_tiles_prefix_sum, problem_tiles_prefix_sum.data(),
                                 cur_problem_count * sizeof(int), cudaMemcpyHostToDevice));
      d_problem_tiles_prefix_sum_vec.push_back(d_problem_tiles_prefix_sum);

      auto kernel     = kernels[kernel_idx];
      auto launch_cfg = launch_cfgs[kernel_idx];

      checkCudaErrors(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, launch_cfg.dynamicSmemBytes));
      checkCudaErrors(cudaLaunchKernelEx(  //
          &launch_cfg, kernel,             //
          ptr_As, ptr_Bs,                  //
          ptr_scale_zp_a,                  //
          ptr_scale_zp_b,                  //
          ptr_Cs, ptr_Ds,                  //
          ldas, ldbs, ldcs, ldds,          //
          problem_sizes,                   //
          qbits_list,                      //
          cur_problem_count,               //
          cur_problem_offset,              //
          d_problem_tiles_prefix_sum       //
          ));
    }
  }

  // checkCudaErrors(cudaDeviceSynchronize());
  for (auto p : d_problem_tiles_prefix_sum_vec)
    checkCudaErrors(cudaFree(p));
  for(auto s : streams)
    checkCudaErrors(cudaStreamDestroy(s));
}
""")