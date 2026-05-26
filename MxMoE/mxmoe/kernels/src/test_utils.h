#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <numeric>
#include <string_view>

#include "fmt/base.h"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "helper.h"  // IWYU pragma: export

using std::string_view;
using thrust::device_vector;
using thrust::host_vector;

using namespace std::string_view_literals;
using namespace std::string_literals;
using namespace std::chrono_literals;

template <typename T>
inline T cdiv(T a, T b) {
  return (a + b - 1) / b;
}

template <>
class fmt::formatter<int2> {
 public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template <typename Context>
  constexpr auto format(int2 const& coord, Context& ctx) const {
    return format_to(ctx.out(), "({},{})", coord.x, coord.y);
  }
};

template <>
class fmt::formatter<dim3> {
 public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template <typename Context>
  constexpr auto format(dim3 const& coord, Context& ctx) const {
    return format_to(ctx.out(), "({},{},{})", coord.x, coord.y, coord.z);
  }
};

inline auto stringfy_launch_config(cudaLaunchConfig_t cfg) {
  return fmt::format("gridDim: {}, blockDim: {}, dyn-smem: {} bytes", cfg.gridDim, cfg.blockDim, cfg.dynamicSmemBytes);
}
template <class T>
constexpr std::string_view type_name() {
  using namespace std;
#ifdef __clang__
  string_view p = __PRETTY_FUNCTION__;
  return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
  string_view p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
  return string_view(p.data() + 36, p.size() - 36 - 1);
#else
  return string_view(p.data() + 49, p.find(';', 49) - 49);
#endif
#elif defined(_MSC_VER)
  string_view p = __FUNCSIG__;
  return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}

template <typename T, typename Y>
auto check_gemm(const host_vector<T>& a, const host_vector<Y>& b, string_view s, float tol = 1.0f) {
  if (!s.empty()) {
    fmt::print("{}: ", s);
  }

  // auto pass = std::equal(begin(a), end(a), begin(b));
  auto pass = true;
  auto len  = a.size();

  auto i = 0;
  for (; i < len; i++) {
    if (fabsf(float(a[i]) - float(b[i])) > tol) {
      pass = false;
      break;
    }
  }

  if (pass) {
    fmt::println("\033[32mcheck pass ...\033[0m");
  } else {
    fmt::println("\033[31mcheck failed!\033[0m");
  }

  return pass ? -1 : i;
}

template <typename F, typename... Args>
auto bench_func(string_view desc, F&& func, bool is_gpu_kernel, int warmup = 30, int bench_iter = 50,
                int sec_at_most = 30) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto t1 = std::chrono::time_point<std::chrono::steady_clock>{};
  auto t2 = std::chrono::time_point<std::chrono::steady_clock>{};

  auto record_st = [&start, &t1, is_gpu_kernel]() {
    if (is_gpu_kernel)
      cudaEventRecord(start);
    else
      t1 = std::chrono::steady_clock::now();
  };
  auto record_ed = [&stop, &t2, is_gpu_kernel]() {
    if (is_gpu_kernel) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
    } else {
      t2 = std::chrono::steady_clock::now();
    }
  };
  auto record_elapsed = [&](float& dura) {
    if (is_gpu_kernel)
      cudaEventElapsedTime(&dura, start, stop);
    else
      dura = std::chrono::duration<float, std::milli>(t2 - t1).count();
  };

  auto WARMUP    = 30;
  auto TEST_ITER = 50;
  auto dura_vec  = host_vector<float>{};
  auto dura      = float{};

  auto real_warmup_iter = 0;
  auto real_test_iter   = 0;
  auto real_warmup_sec  = 0.f;
  auto real_test_sec    = 0.f;

  // 1. warmup
  auto launch_begin = std::chrono::steady_clock::now();
  for (auto i = 0; i < WARMUP; i++) {
    func();
    cudaDeviceSynchronize();
    auto launch_continue = std::chrono::steady_clock::now();

    real_warmup_sec = std::chrono::duration<float, std::milli>(launch_continue - launch_begin).count() / 1e3;
    real_warmup_iter += 1;

    if (real_warmup_sec > (sec_at_most / 3.0f)) break;
  }

  // 2. benchmark
  for (auto i = 0; i < TEST_ITER; i++) {
    record_st();
    func();
    record_ed();
    record_elapsed(dura);

    real_test_iter += 1;
    real_test_sec += dura / 1000;
    if (real_test_sec > sec_at_most) break;

    dura_vec.push_back(dura);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::sort(begin(dura_vec), end(dura_vec));

  auto collect_len = dura_vec.size();
  auto max_time    = dura_vec.back();
  auto min_time    = dura_vec.front();
  auto median      = dura_vec[collect_len / 2];

  auto avg_time = std::reduce(begin(dura_vec) + 3, begin(dura_vec) + collect_len - 3, 0.000000001f);
  avg_time /= (collect_len - 6);

  return std::make_pair(                                                     //
      fmt::format(                                                           //
          "\033[1;34m>>>BENCH {}\033[0m\n"                                   //
          "    warmup-iter: {}({:.3f} s), test-iter: {}({:.3f} s)\n"         //
          "{:^15}    {:^15}    {:^15}    {:^15}\n"                           //
          "{:^15.4f}    {:^15.4f}    {:^15.4f}    {:^15.4f}\n",              //
          desc,                                                              //
          real_warmup_iter, real_warmup_sec, real_test_iter, real_test_sec,  //
          "max(ms)", "min(ms)", "avg(ms)", "meadian(ms)",                    //
          max_time, min_time, avg_time, median                               //
          ),                                                                 //
      median                                                                 //
  );
}

template <bool Trans = false, typename T>
void dump_arr(T* arr, int ldm, std::pair<int, int> tile_size, string_view desc = ""sv, _IO_FILE* fp = stdout) {
  if (!desc.empty()) {
    fmt::println("{}", desc);
  }
  auto [m, n] = tile_size;
  if constexpr (!Trans) {
    for (auto i = 0; i < m; i++) {
      for (auto j = 0; j < n; j++) {
        // if (i * n + j < tile_size) {
        fmt::print(fp, "{:<9.3f}, ", float(arr[i * ldm + j]));
        // }
      }
      fmt::println(fp, "");
    }
  } else {
    for (auto i = 0; i < m; i++) {
      for (auto j = 0; j < n; j++) {
        // if (i * n + j < tile_size) {
        fmt::print(fp, "{:<9.3f}, ", float(arr[j * ldm + i]));
        // }
      }
      fmt::println(fp, "");
    }
  }
  fmt::println(fp, "");
}

enum class CaseMode { CHECK, BENCH };

template <typename F, typename Res, typename Ref>
auto check_or_bench_gemm(                                                             //
    CaseMode mode, string_view desc, bool is_gpu_kernel, std::array<int, 3>&& mnk,    //
    F&& kernel, Res&& get_res, Ref&& get_ref,                                         //
    bool verbose = true, int warm_up = 30, int bench_iter = 50, int sec_at_most = 30  //
) {
  auto M = mnk[0], N = mnk[1], K = mnk[2];
  double FLOPS = 2.0 * M * N * K;

  auto mode_str   = mode == CaseMode::CHECK ? "CHECK"sv : "BENCH"sv;
  auto print_info = fmt::format("\033[1;34m>>>\033[0m {} {}", mode_str, desc);

  std::pair<std::pair<double, double>, std::string> bench_res;
  if (mode == CaseMode::CHECK) {
    kernel();
    const auto& res = get_res();
    const auto& ref = get_ref();
    auto pos        = check_gemm(ref, res, print_info);
    if (pos != -1) {
      fmt::println("pos: {} {}", pos, std::make_pair(pos / N, pos % N));
      dump_arr(ref.data() + pos, N, {8, 8}, "ref");
      dump_arr(res.data() + pos, N, {8, 8}, "res");
    }
    bench_res = std::make_pair(std::make_pair(double(0), 0), std::string(desc));
  } else {
    auto [info, avg_time] = bench_func(desc, kernel, is_gpu_kernel, warm_up, bench_iter, sec_at_most);
    double bench_FLOPS    = FLOPS / avg_time * 1e3 / 1e12;
    if (verbose) fmt::println("{}  TFLOPS: {}\n", info, bench_FLOPS);
    bench_res = std::make_pair(std::make_pair(double(avg_time), bench_FLOPS), std::string(desc));
  }
  return bench_res;
}

template <typename Tile>
auto tag_name() {
  return fmt::format("({}, {}, {}), ({}, {}, {}), {}", Tile::BM, Tile::BN, Tile::BK, Tile::WARP_M, Tile::WARP_N,
                     Tile::WARP_K, Tile::STAGE);
}

template <typename... Tiles>
auto tag_name(string_view s) {
  return fmt::format("{}: [{}]", s, (fmt::format("{}; ", tag_name<Tiles>()) + ...));
}

template <typename... Tiles>
auto tag_name(std::string_view s, const std::tuple<Tiles...>& t) {
  return std::apply([&s](auto&&... args) { return tag_name<Tiles...>(s); }, t);
}

#define ErrorInfo(msg) std::runtime_error(fmt::format("{}:{}: {}", __FILE__, __LINE__, msg))
