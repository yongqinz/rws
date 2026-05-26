import os
import argparse
import regex as re
import dataclasses

from enum import Enum
class StrEnum(str, Enum):
    pass
from pprint import pprint
from string import Template
from functools import partial
from itertools import product
from collections import OrderedDict

from project_config import *
from mxmoe.kernels.tile_config import get_gpu_info, TileConfig, get_possible_tile_list, QCFG_MAP, SUPPORTED_QCFG, get_info_from_qcfg_str

class KernelType(StrEnum):
    Fused = "hz_fused"
    Sequential = "seq_launch"
    MultiStream = "ms_launch"


CTA_KERNEL_MAP = {
    **dict.fromkeys(
        ["fp16", "fp16_accfp16", "bf16"],
        "cta_gemm_multistage_v2<TileCfg>(tile_a, tile_b, tile_c, smem, K, lane, warp_x, warp_y, warp_z)"
    ),
    **{
        k: "cta_gemm_multistage_qb_v2<TileCfg>(tile_a, tile_b, tile_c, smem, gmem_scale, smem_scale, N, K, lane, warp_x, warp_y, warp_z, tidx)"
        for k in [f"w{wbits}a16_g-1_{sym}{ext}" for wbits in [2,4,8] for sym in ["sym", "asym"] for ext in ["", "_accfp16"]]
    },
    **{
        k: "cta_gemm_wxa16g128<TileCfg>(tile_a, tile_b, tile_c, smem, gmem_scale, smem_scale, N, K, lane, warp_x, warp_y, warp_z, tidx)"
        for k in [f"w{wbits}a16_g128_{sym}{ext}" for wbits in [2,4,8] for sym in ["sym", "asym"] for ext in ["", "_accfp16"]]
    },
    **dict.fromkeys(
        ["w8a8_g-1_sym", "w8a8_g-1_sym_e4m3", "w4a4_g-1_sym"],
        "cta_gemm_multistage_qab_v2<TileCfg>(tile_a, tile_b, tile_c, smem, gmem_scale_a, gmem_scale_b, smem_scale_a, smem_scale_b, M, N, K, lane, warp_x, warp_y, warp_z, tidx)"
    ),
    "w4a4_g128_sym": "cta_gemm_w4a4g128<TileCfg>(tile_a, tile_b, tile_c, smem, gmem_scale_a, gmem_scale_b, smem_scale_a, smem_scale_b, M, N, K, lane, warp_x, warp_y, warp_z, tidx)",
}

def _gen_condition(qcfg: str):
    w_bits, a_bits, gsize, sym = get_info_from_qcfg_str(qcfg)
    sym = "true" if sym else "false"
    return f"(a_bits == {a_bits} && w_bits == {w_bits} && gsize == {gsize} && sym == {sym})"

BRANCH_CONDITION_MAP = {
    **dict.fromkeys(["fp16", "fp16_accfp16", "bf16"], "(a_bits == 16 && w_bits == 16)"),
    **{k: _gen_condition(k) for k in ["w4a4_g-1_sym","w4a4_g128_sym", "w8a8_g-1_sym"]},
    **{k: _gen_condition(k) for k in [
        f"w{wbits}a16_g{gsize}_{sym}{ty}"
        for wbits in [8,4,2]
        for gsize in [-1, 128]
        for sym in ["sym", "asym"]
        for ty in ["", "_accfp16", "_bf16"]
    ]},
}


def format_template(tile_size: TileConfig, quant_tile_cfg: TileConfig):
    return TileConfig(
                BM=tile_size.BM, BN=tile_size.BN, BK=tile_size.BK,
                WM=tile_size.WM, WN=tile_size.WN, WK=tile_size.WK,
                STAGE=tile_size.STAGE, SPLITK=tile_size.SPLITK,
                MMA=tile_size.MMA,
                QCFGA=quant_tile_cfg.QCFGA, QCFGB=quant_tile_cfg.QCFGB
            )

def is_fusion_compatible(tile_cfgs: list[TileConfig]):
    x = [cfg.num_warps for cfg in tile_cfgs]
    return all(tile_cfgs[0].num_warps == cfg.num_warps for cfg in tile_cfgs)


def replace_target_regex(template, target, value):
    pattern = re.compile(target)
    result = pattern.sub(value, template)
    return result

class TemplateGenerator:
    def __init__(self, arch: str, qcfgs: list[str], kernel_type: KernelType, tile_configs: dict[str, list[TileConfig]]=None):
        self.arch = arch
        self.qcfgs = sorted(qcfgs)
        self.kernel_type = kernel_type
        self.tile_configs = tile_configs

        
    def get_tile_configs(self):
        if self.tile_configs is not None:
            tile_configs = OrderedDict({
                qcfg: self.tile_configs[qcfg] for qcfg in self.qcfgs
            })
        else:
            tile_configs = OrderedDict({
                qcfg: [format_template(tile_size, QCFG_MAP[qcfg]) for tile_size in get_possible_tile_list(self.arch, qcfg)]
                        for qcfg in self.qcfgs
            })

        keys = tile_configs.keys()
        if self.kernel_type == KernelType.Fused:
            combinations = filter(lambda x: is_fusion_compatible(x), product(*tile_configs.values()))
        else:
            combinations = product(*tile_configs.values())
        candidate_templates = [OrderedDict(zip(keys, comb)) for comb in combinations]
        # for r in candidate_templates:
        #     # print({k: v.to_str() for k, v in r.items()})
        #     print("template<{}>".format(", ".join([v.to_str() for v in r.values()])))
        # print(f"len(candidate_templates): {len(candidate_templates)}")
        return candidate_templates


    def cvt_qcfg_str_to_condition(self, qcfg: str):
        assert qcfg in self.qcfgs, f"Invalid qcfg: {qcfg}"

        # TODO: convert to a id to identify the qcfg
        return BRANCH_CONDITION_MAP[qcfg]


    def get_smem_size(self, tile_cfg_dict: OrderedDict[str, TileConfig]):
        '''
        in bytes
        '''
        if self.kernel_type == KernelType.Fused:
            # 1. smem_tile: maximum of all tile configurations
            smem_tile = max([tile_cfg_dict[qcfg].smem_bytes_tile() for qcfg in self.qcfgs])
            # 2. smem_scale: maximum of all tile configurations
            smem_scale = max([tile_cfg_dict[qcfg].smem_bytes_scale() for qcfg in self.qcfgs])
        else:
            assert len(tile_cfg_dict) == 1, "Sequential kernel only supports one qcfg"
            smem_tile = next(iter(tile_cfg_dict.values())).smem_bytes_tile()
            smem_scale = next(iter(tile_cfg_dict.values())).smem_bytes_scale()

        return smem_tile, smem_scale


    def build_selection_branch(self):
        selection_block = []
        for i, qcfg in enumerate(self.qcfgs):
            b = "{keyword} {cond} {body}".format(
                keyword="else if" if i > 0 else "if",
                cond=self.cvt_qcfg_str_to_condition(qcfg),
                body=f"${{branch{i}}}"
            )
            selection_block.append(b)

        branch_block = " ".join(selection_block)

        return branch_block


    def build_schedule(self, tile_cfg_dict: OrderedDict[str, TileConfig]):
        smem_size_tile, smem_size_scale = self.get_smem_size(tile_cfg_dict)

        template = Template("""
        {
        using TA = typename MatricesInfo::TA;
        using TB = typename MatricesInfo::TB;
        using TC = typename MatricesInfo::TC;
        static_assert(std::is_same_v<TA, TB> && std::is_same_v<TA, half>, "");
        constexpr auto LAYOUT_A = MatricesInfo::LAYOUT_A;
        constexpr auto LAYOUT_B = MatricesInfo::LAYOUT_B;
        constexpr auto LAYOUT_C = MatricesInfo::LAYOUT_C;

        int lane = threadIdx.x;
        int tidx = lane + threadIdx.y * WarpSize;

        extern __shared__ uint8_t smem[];
        ${smem_alloc}

        auto visitor  = TileScheduler(problem_sizes, problem_count, problem_tiles_prefix_sum);
        auto tile_idx = blockIdx.x;
        auto cta_size = gridDim.x;

        while (true) {
            // get corresponding [tileA, tileB, tileC] offset
            auto problem_idx = visitor.get_problem_idx(tile_idx);
            // early exit if all problems are done
            if (problem_idx == -1) break;

            // [act, w]
            int a_bits = qbits_list[problem_idx].qbits.x;
            int w_bits = qbits_list[problem_idx].qbits.y;
            int gsize  = qbits_list[problem_idx].gsize;
            bool sym   = qbits_list[problem_idx].sym;

            ${tile_coord}
            
            auto problem_size = problem_sizes[problem_idx];
            auto M            = problem_size.x;
            auto N            = problem_size.y;
            auto K            = problem_size.z;

            ${branch_block}

            // advance
            tile_idx += cta_size;
        }
        }
        """)

        branch_block = self.build_selection_branch()

        
        tile_coord = Template("""
            // get coordinate of TileC of current problem
            auto tile_coord = [&] {
              ${branch_block}
              return dim3(0, 0, 0);
            }();
        """).substitute(branch_block=branch_block)

        
        tile_coord = Template(tile_coord).substitute({
            f"branch{i}": f"return visitor.get_tile_coord<{tile_cfg_dict[self.qcfgs[i]].BM}, {tile_cfg_dict[self.qcfgs[i]].BN}>(problem_idx, tile_idx);\n" for i in range(len(self.qcfgs))
        })

        template = Template(template.substitute(
            {
                "smem_alloc": f"auto smem_scale_zp = reinterpret_cast<half*>(smem + {smem_size_tile});",
                "tile_coord": tile_coord,
                "branch_block": branch_block
            }
        ))

        return template

    
    def build_impl_branch_body(self, tile_cfg_dict: OrderedDict[str, TileConfig], impl_body: Template):
        def build_one_branch(i: int):
            tile_cfg: TileConfig = tile_cfg_dict[self.qcfgs[i]]

            body = Template("""{
              using TileCfg = ${TileConfig};
              using TileA = ${TileA};
              using TileB = ${TileB};
              using TileC = ${TileC};

              auto warp_x = TileCfg::warp_x();
              auto warp_y = TileCfg::warp_y();
              auto warp_z = TileCfg::warp_z();

              auto tile_a = TileA(ptr_As[problem_idx], M, K, tile_coord.y, tile_coord.x, warp_y, warp_z, tidx);
              auto tile_b = TileB(ptr_Bs[problem_idx], N, K, tile_coord.y, tile_coord.x, warp_x, warp_z, tidx);
              auto tile_c = TileC(ptr_Cs[problem_idx], M, N, tile_coord.y, tile_coord.x, tidx);

              ${context}

              ${func_call};
            }
            """)

            tile_cfg_str = tile_cfg.to_str()

            if not tile_cfg.is_a_quant():
                tile_a = f"GemmTileA<TA, LAYOUT_A, TileCfg>"
            else:
                tile_a = f"QuantTileA<half, LAYOUT_A, TileCfg, TileCfg::QConfigA>"

            if not tile_cfg.is_b_quant():
                tile_b = f"GemmTileB<TB, LAYOUT_B, TileCfg>"
            else:
                tile_b = f"QuantTileB<half, LAYOUT_B, TileCfg, TileCfg::QConfigB>"

            if tile_cfg.is_a_quant() and tile_cfg.is_b_quant():
                context = """
                auto smem_scale_a = smem_scale_zp;
                auto smem_scale_b = smem_scale_zp + TileA::CTA_SCALE_SIZE;

                auto gmem_scale_a = ptr_scale_zp_a[problem_idx] + tile_coord.x * TileA::CTA_SCALE_SIZE;
                auto gmem_scale_b = ptr_scale_zp_b[problem_idx] + tile_coord.y * TileB::CTA_SCALE_SIZE;
                """
            elif tile_cfg.is_b_quant():
                context = """
                auto smem_scale = smem_scale_zp;
                auto gmem_scale = ptr_scale_zp_b[problem_idx] + tile_coord.y * TileB::CTA_SCALE_SIZE;
                """
            else:
                context = ""

            ret = body.substitute(
                {
                    "TileA": tile_a,
                    "TileB": tile_b,
                    "TileC": "GlobalTileC<TC, LAYOUT_C, TileCfg>",
                    "TileConfig": tile_cfg_str,
                    "context": context,
                    "func_call": CTA_KERNEL_MAP[self.qcfgs[i]],
                }
            )
            return ret

        return impl_body.substitute(
            {
                f"branch{i}": build_one_branch(i) for i in range(len(self.qcfgs))
            }
        )
            
    
    def build_single_type_impl_body(self, tile_cfg_dict: OrderedDict[str, TileConfig], qcfg: str):
        smem_size_tile, smem_size_scale = self.get_smem_size(tile_cfg_dict)

        template = Template("""
        {
        using TA = typename MatricesInfo::TA;
        using TB = typename MatricesInfo::TB;
        using TC = typename MatricesInfo::TC;
        static_assert(std::is_same_v<TA, TB> && std::is_same_v<TA, half>, "");
        constexpr auto LAYOUT_A = MatricesInfo::LAYOUT_A;
        constexpr auto LAYOUT_B = MatricesInfo::LAYOUT_B;
        constexpr auto LAYOUT_C = MatricesInfo::LAYOUT_C;

        ptr_As += problem_offset;
        ptr_Bs += problem_offset;
        ptr_scale_zp_a += problem_offset;
        ptr_scale_zp_b += problem_offset;
        ptr_Cs += problem_offset;
        ptr_Ds += problem_offset;
        problem_sizes += problem_offset;

        qbits_list += problem_offset;
        int lane = threadIdx.x;
        int tidx = lane + threadIdx.y * WarpSize;

        extern __shared__ uint8_t smem[];
        ${smem_alloc}

        auto visitor  = TileScheduler(problem_sizes, problem_count, problem_tiles_prefix_sum);
        auto tile_idx = blockIdx.x;
        auto cta_size = gridDim.x;

        using TileCfg = ${TileConfig};
        using TileA = ${TileA};
        using TileB = ${TileB};
        using TileC = ${TileC};

        auto warp_x = TileCfg::warp_x();
        auto warp_y = TileCfg::warp_y();
        auto warp_z = TileCfg::warp_z();

        while (true) {
            // get corresponding [tileA, tileB, tileC] offset
            auto problem_idx = visitor.get_problem_idx(tile_idx);
            // early exit if all problems are done
            if (problem_idx == -1) break;

            auto tile_coord = visitor.get_tile_coord<${BMN}>(problem_idx, tile_idx);
            
            auto problem_size = problem_sizes[problem_idx];
            auto M            = problem_size.x;
            auto N            = problem_size.y;
            auto K            = problem_size.z;

            auto tile_a = TileA(ptr_As[problem_idx], M, K, tile_coord.y, tile_coord.x, warp_y, warp_z, tidx);
            auto tile_b = TileB(ptr_Bs[problem_idx], N, K, tile_coord.y, tile_coord.x, warp_x, warp_z, tidx);
            auto tile_c = TileC(ptr_Cs[problem_idx], M, N, tile_coord.y, tile_coord.x, tidx);

            ${context}

            ${func_call};

            // cta_gemm_multistage_v2<TileConfig>(tile_a, tile_b, tile_c, shmem, K, lane, warp_x, warp_y, warp_z);

            // advance
            tile_idx += cta_size;
        }
        }
        """)


        tile_cfg = list(tile_cfg_dict.values())[0]

        if not tile_cfg.is_a_quant():
            tile_a = f"GemmTileA<TA, LAYOUT_A, TileCfg>"
        else:
            tile_a = f"QuantTileA<half, LAYOUT_A, TileCfg, TileCfg::QConfigA>"

        if not tile_cfg.is_b_quant():
            tile_b = f"GemmTileB<TB, LAYOUT_B, TileCfg>"
        else:
            tile_b = f"QuantTileB<half, LAYOUT_B, TileCfg, TileCfg::QConfigB>"

        if tile_cfg.is_a_quant() and tile_cfg.is_b_quant():
            context = """
            auto smem_scale_a = smem_scale_zp;
            auto smem_scale_b = smem_scale_zp + TileA::CTA_SCALE_SIZE;

            auto gmem_scale_a = ptr_scale_zp_a[problem_idx] + tile_coord.x * TileA::CTA_SCALE_SIZE;
            auto gmem_scale_b = ptr_scale_zp_b[problem_idx] + tile_coord.y * TileB::CTA_SCALE_SIZE;
            """
        elif tile_cfg.is_b_quant():
            context = """
            auto smem_scale = smem_scale_zp;
            auto gmem_scale = ptr_scale_zp_b[problem_idx] + tile_coord.y * TileB::CTA_SCALE_SIZE;
            """
        else:
            context = ""

        return template.substitute(
            {
                "smem_alloc": f"auto smem_scale_zp = reinterpret_cast<half*>(smem + {smem_size_tile});",
                "TileConfig": tile_cfg.to_str(),
                "TileA": tile_a,
                "TileB": tile_b,
                "TileC": "GlobalTileC<TC, LAYOUT_C, TileCfg>",
                "BMN": f"{tile_cfg.BM}, {tile_cfg.BN}",
                "context": context,
                "func_call": CTA_KERNEL_MAP[qcfg],
            }
        )


    def build_impl_body(self, tile_cfg_dict: OrderedDict[str, TileConfig], qcfg: str=None):
        if self.kernel_type == KernelType.Fused:
            impl_body = self.build_schedule(tile_cfg_dict)
            impl_body = self.build_impl_branch_body(tile_cfg_dict, impl_body)
        else:
            impl_body = self.build_single_type_impl_body(tile_cfg_dict, qcfg)

        return impl_body
    

    def build_api_body(self, tile_cfg_dict: OrderedDict[str, TileConfig], kernel_name: str):
        from mxmoe.kernels.kernel_sketch import HZ_FUSE_LAUNCH_TEMPLATE, SEQUENTIAL_LAUNCH_TEMPLATE, MULTI_STREAM_LAUNCH_TEMPLATE

        if self.kernel_type == KernelType.Fused:
            tile_cfg_template_str = ", ".join([v.to_str() for v in tile_cfg_dict.values()])

            return HZ_FUSE_LAUNCH_TEMPLATE.substitute({
                "KernelName": kernel_name,
                "TileConfigs": tile_cfg_template_str,
                "tile_counts": Template(self.build_selection_branch()).substitute({
                    f"branch{i}": f"total_tiles+=cu_cdiv(M,{tile_cfg_dict[qcfg].BM})*cu_cdiv(N,{tile_cfg_dict[qcfg].BN});"
                    for i, qcfg in enumerate(self.qcfgs)
                })+"""else throw std::runtime_error("quant type not supported");""",
                "smem_size": sum(self.get_smem_size(tile_cfg_dict)),
                "num_warps": tile_cfg_dict[self.qcfgs[0]].num_warps,
            })
        elif self.kernel_type == KernelType.Sequential:
            kernel_array = f"[{len(tile_cfg_dict)}]"
            kernel_array += "{{{}}};".format(",".join(f"&{kernel_name}_{i}_impl" for i in range(len(tile_cfg_dict))))
            launch_cfg_array = []
            for t in tile_cfg_dict.values():
                smem = t.smem_bytes_tile()+ t.smem_bytes_scale()
                num_warps = t.num_warps
                launch_cfg_array.append(
                f"cudaLaunchConfig_t{{.gridDim = dim3(num_ctas),.blockDim=dim3(32, {num_warps}), .dynamicSmemBytes = {smem}}}"
                )
            launch_cfg_array = f"[{len(tile_cfg_dict)}]" + "{{{}}};".format(",".join(launch_cfg_array))
            return SEQUENTIAL_LAUNCH_TEMPLATE.substitute({
                "KernelName": kernel_name,
                "KernelArray": kernel_array,
                "LaunchConfigArray": launch_cfg_array,
                "tile_counts": Template(self.build_selection_branch()).substitute({
                    f"branch{i}": f"total_tiles+=cu_cdiv(M,{tile_cfg_dict[qcfg].BM})*cu_cdiv(N,{tile_cfg_dict[qcfg].BN});"
                    for i, qcfg in enumerate(self.qcfgs)
                })+"else throw std::runtime_error(\"quant type not supported\");\nproblem_tiles_prefix_sum.push_back(total_tiles);",
            })
        else:
            kernel_array = f"[{len(tile_cfg_dict)}]"
            kernel_array += "{{{}}};".format(",".join(f"&{kernel_name}_{i}_impl" for i in range(len(tile_cfg_dict))))
            stream_array = f"[{len(tile_cfg_dict)}];\n"
            stream_array += "{}".format("\n".join(f"cudaStreamCreate(&streams[{i}]);" for i in range(len(tile_cfg_dict))))
            launch_cfg_array = []
            for i, t in enumerate(tile_cfg_dict.values()):
                smem = t.smem_bytes_tile()+ t.smem_bytes_scale()
                num_warps = t.num_warps
                launch_cfg_array.append(
                f"cudaLaunchConfig_t{{.gridDim = dim3(num_ctas),.blockDim=dim3(32,{num_warps}),.dynamicSmemBytes={smem},.stream=streams[{i}]}}"
                )
            launch_cfg_array = f"[{len(tile_cfg_dict)}]" + "{{{}}};".format(",".join(launch_cfg_array))
            return MULTI_STREAM_LAUNCH_TEMPLATE.substitute({
                "KernelName": kernel_name,
                "StreamArray": stream_array,
                "KernelArray": kernel_array,
                "LaunchConfigArray": launch_cfg_array,
                "tile_counts": Template(self.build_selection_branch()).substitute({
                    f"branch{i}": f"total_tiles+=cu_cdiv(M,{tile_cfg_dict[qcfg].BM})*cu_cdiv(N,{tile_cfg_dict[qcfg].BN});"
                    for i, qcfg in enumerate(self.qcfgs)
                })+"else throw std::runtime_error(\"quant type not supported\");\nproblem_tiles_prefix_sum.push_back(total_tiles);",
            })


    def build_single_kernel(self, tile_cfg: OrderedDict[str, TileConfig], kname: str):
        from mxmoe.kernels.kernel_sketch import KERNEL_TEMPLATE, API_TEMPLATE

        tile_cfg_template_str = ", ".join([v.to_str() for v in tile_cfg.values()])
        if self.kernel_type == KernelType.Fused:
            decl_kernel = KERNEL_TEMPLATE.substitute({
                "KernelName": kname,
                "TileConfigs": tile_cfg_template_str,
                "ProblemOffset": "",
                "ImplBody": ";"
            })
            defn_kernel = KERNEL_TEMPLATE.substitute({
                "KernelName": kname,
                "TileConfigs": tile_cfg_template_str,
                "ProblemOffset": "",
                "ImplBody": self.build_impl_body(tile_cfg),
            })
        else:
            knames = [f"{kname}_{i}" for i in range(len(tile_cfg))]
            kname_2_tile_cfg = {k: OrderedDict({t: tile_cfg[t]}) for k, t in zip(knames, tile_cfg.keys())}
            decl_kernel = "\n\n".join(KERNEL_TEMPLATE.substitute({
                "KernelName": kname_,
                "TileConfigs": "",
                "ProblemOffset": "int problem_offset,//",
                "ImplBody": ";"
            }) for kname_ in knames) 
            defn_kernel = "\n\n".join(KERNEL_TEMPLATE.substitute({
                "KernelName": kname_,
                "TileConfigs": "",
                "ProblemOffset": "int problem_offset,//",
                "ImplBody": self.build_impl_body(kname_2_tile_cfg[kname_], qcfg),
            }) for qcfg, kname_ in zip(self.qcfgs, knames))
        decl_api = API_TEMPLATE.substitute({
            "KernelName": kname,
            "TileConfigs": tile_cfg_template_str,
            "APIBody": ";"
        })
        defn_api = API_TEMPLATE.substitute({
            "KernelName": kname,
            "TileConfigs": "",
            "APIBody": self.build_api_body(tile_cfg, kname),
        })
        reg = f"struct Register_{kname}{{Register_{kname}(){{register_kernel<{tile_cfg_template_str}>(&{kname},\"{kname}\");}}}};"
        reg += f"static Register_{kname} register_{kname};"

        decl = decl_kernel + "\n\n" + decl_api
        defn = defn_kernel + "\n\n" + defn_api
        return decl, defn, reg


    def generate_source_code(self):
        from mxmoe.kernels.kernel_sketch import KERNEL_DECLARATION_TEMPLATE, KERNEL_DEFINATION_TEMPLATEE

        tile_cfgs = self.get_tile_configs()
        # print(tile_cfgs)

        kernel_decl = []
        kernel_defs = []
        registers = []
        for i, tile_cfg in enumerate(tile_cfgs):
            # if i > 2: break
            kname = f"groupgemm_{self.kernel_type}_{i}"

            decl, defn, reg = self.build_single_kernel(tile_cfg, kname)

            kernel_decl.append(decl)
            kernel_defs.append(defn)
            registers.append(reg)

            # print(kernel_defs[0])

        kernel_header = KERNEL_DECLARATION_TEMPLATE.substitute({
            "Signatures": "\n".join(kernel_decl),
        })
        out_dir = f"{os.path.dirname(__file__)}/src/generated"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(f"{out_dir}/{self.kernel_type}_decl.cuh", "w") as f:
            f.write(kernel_header)

        for i, kernel_def in enumerate(kernel_defs):
            kernel_src = KERNEL_DEFINATION_TEMPLATEE.substitute({
                "KernelType": self.kernel_type,
                "Definitions": kernel_def,
                "Register": registers[i],
            })
            with open(f"{out_dir}/{self.kernel_type}_{i}.cu", "w") as f:
                f.write(kernel_src)

        print(f"Generated {len(tile_cfgs)} kernels for `{self.kernel_type}` {self.qcfgs}")
        # print(kernel_decl)
        # print(source_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose kernel")
    parser.add_argument("--type", type=str, default="Fused", choices=["Fused", "Sequential", "MultiStream"], help="Kernel type")
    parser.add_argument("--qcfgs", type=str, nargs="+", default=["fp16"], help="Quantization configurations")

    args = parser.parse_args(
        # [
        #     "--type", "Sequential",
        #     "--num", "1"
        # ]
    )

    for qcfg in args.qcfgs:
        assert qcfg in SUPPORTED_QCFG, f"Unsupported qcfg: {qcfg}"

    gpu_info = get_gpu_info()
    print(f"GPU Info: {gpu_info}")
    # generator = TemplateGenerator(gpu_info["cc"], args.qcfgs, KernelType.Fused)
    generator = TemplateGenerator("89", args.qcfgs, KernelType.Fused)
    # generator = TemplateGenerator(gpu_info["cc"], ["w4a4_g-1_sym", "w8a8_g-1_sym"], KernelType.Fused)
    generator.generate_source_code()

    # generator = TemplateGenerator(cc, ["w4a16_g-1_asym", "w8a8_g-1_sym"], KernelType.Sequential)
    # generator.generate_source_code()
    # generator = TemplateGenerator(cc, ["w4a16_g-1_asym", "w8a8_g-1_sym"], KernelType.MultiStream)
    # generator.generate_source_code()

    