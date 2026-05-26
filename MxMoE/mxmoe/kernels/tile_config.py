import dataclasses

def get_gpu_info():
    import pynvml
    import pycuda.driver as cuda
    import pycuda.autoinit  # This initializes CUDA driver

    res = {}

    pynvml.nvmlInit()
    device_count = cuda.Device.count()
    for i in range(device_count):
        device = cuda.Device(i)
        sm_count = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
        compute_capability = device.compute_capability()
        max_shared_memory_per_mp = device.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)

        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        cur_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        max_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)


        res["cc"] = "".join([str(x) for x in compute_capability])
        res["sm_count"] = sm_count
        res["frequency"] = (cur_clock, max_clock)
        res["max_shared_memory_per_mp"] = max_shared_memory_per_mp
        break
    pynvml.nvmlShutdown()

    return res

TYPE_TO_BITS = {
    **dict.fromkeys(["half", "bfloat16"],16),
    **dict.fromkeys(["float", "int32_t"],32),
    **dict.fromkeys(["int8_t", "uint8_t"],8),
    **dict.fromkeys(["int4_t", "uint4_t"],4),
    **dict.fromkeys(["int2_t", "uint2_t"],2),
}

SUPPORTED_QCFG = [
    "fp16", "fp16_accfp16",
    "bf16",

    "w8a8_g-1_sym",
    "w8a8_g-1_sym_E4M3", # FP8

    "w4a4_g-1_sym",
    "w4a4_g128_sym",

    *[
        f"w{wbits}a16_g{gsize}_{sym}{ty}"
        for wbits in [8,4,2]
        for gsize in [-1, 128]
        for sym in ["sym", "asym"]
        for ty in ["", "_accfp16", "_bf16"]
    ],

    # "w1a16_g128",
]


MMA_SHAPE = {
    "MMA_FP16_FP32": (16, 8, 16),
    "MMA_FP16_FP16": (16, 8, 16),
    "MMA_BF16_FP32": (16, 8, 16),
    "MMA_E4M3_K32": (16, 8, 32),
    "MMA_S8_K32": (16, 8, 32),
    "MMA_S4_K64": (16, 8, 64),
}

MMA_TO_ACC_TYPE = {
    "MMA_FP16_FP32": "float",
    "MMA_FP16_FP16": "half",
    "MMA_BF16_FP32": "float",
    "MMA_E4M3_K32": "float",
    "MMA_S8_K32": "int32_t",
    "MMA_S4_K64": "int32_t"
}
MMA_TO_INP_TYPE = {
    "MMA_FP16_FP32": "half",
    "MMA_FP16_FP16": "half",
    "MMA_BF16_FP32": "bfloat16",
    "MMA_E4M3_K32": "fp8e4m3",
    "MMA_S8_K32": "int8_t",
    "MMA_S4_K64": "int4_t"
}

QCFG_TO_MMA = {
    **dict.fromkeys([
        "fp16",
        *[f"w{wbits}a16_g{gsize}_{sym}" for wbits in [8, 4, 2] for gsize in [-1, 128] for sym in ["sym", "asym"]],
    ], "MMA_FP16_FP32"),

    **dict.fromkeys([
        "fp16_accfp16",
        *[f"w{wbits}a16_g{gsize}_{sym}_accfp16" for wbits in [8, 4, 2] for gsize in [-1, 128] for sym in ["sym", "asym"]],
    ], "MMA_FP16_FP16"),

    **dict.fromkeys([
        "bf16",
        *[f"w{wbits}a16_g{gsize}_{sym}_bf16" for wbits in [8, 4, 2] for gsize in [-1, 128] for sym in ["sym", "asym"]],
    ], "MMA_BF16_FP32"),

    **dict.fromkeys([
        "w8a8_g-1_sym_E4M3",
    ], "MMA_E4M3_K32"),

    **dict.fromkeys([
        "w8a8_g-1_sym",
    ], "MMA_S8_K32"),

    **dict.fromkeys([
        "w4a4_g-1_sym", "w4a4_g128_sym",
    ], "MMA_S4_K64"),
}


CFG_TEMPLATE = "TileConfig<{BM},{BN},{BK},{WM},{WN},{WK},{STAGE},{SPLITK},{MMA},{QCFGA},{QCFGB}>" 

# TODO: now fix the type of scale to `half`
@dataclasses.dataclass
class QConfigBase:
    T_PACK: str="half"
    QBITS: int=16
    GSIZE: int=-1
    SYM: bool=True
    PACK_DIM: str="PackDim::K"
    USE_FP: bool=False
    T_SCALE: str="half"

    def bytes_per_scale_zp(self):
        raise NotImplementedError
    
    # TODO: assume packed as 16-bit dtype, and 16 % QBITS == 0
    def pack_num(self):
        return 16 // self.QBITS
    
    def qtype(self) -> str:
        if self.QBITS >= 16:
            return None 
        else:
            if self.QBITS == 8 and self.USE_FP:
                return "fp8e4m3"
            elif self.QBITS == 8:
                assert self.USE_FP == False
                return "int8_t" if self.SYM else "uint8_t"
            elif self.QBITS == 4:
                assert self.USE_FP == False
                return "int4_t" if self.SYM else "uint4_t"
            elif self.QBITS == 2:
                assert self.USE_FP == False
                return "int2_t" if self.SYM else "uint2_t"
            else:
                raise ValueError(f"Unsupported QBITS: {self.QBITS}")


@dataclasses.dataclass
class NO_QUANT(QConfigBase):
    def to_str(self):
        return "NO_QUANT"

    def bytes_per_scale_zp(self):
        return 0


@dataclasses.dataclass
class QConfig(QConfigBase):
    def to_str(self):
        return f"QConfig<{self.T_PACK}, {'true' if self.SYM else 'false'}, {self.QBITS}, {self.GSIZE}, {self.PACK_DIM}, {'true' if self.USE_FP else 'false'}, {self.T_SCALE}>"

    def bytes_per_scale_zp(self):
        return TYPE_TO_BITS[self.T_SCALE]//8 * (1 if self.SYM else 2)


QCFG_W8A8=QConfig(T_PACK="half", QBITS=8, GSIZE=-1, SYM=True, PACK_DIM="PackDim::K", USE_FP=False, T_SCALE="half")
QCFG_W4A4=QConfig(T_PACK="half", QBITS=4, GSIZE=-1, SYM=True, PACK_DIM="PackDim::K", USE_FP=False, T_SCALE="half")
QCFG_W4A4_G128=QConfig(T_PACK="half", QBITS=4, GSIZE=128, SYM=True, PACK_DIM="PackDim::K", USE_FP=False, T_SCALE="half")

QCFG_W8A16_SYM=QConfig(T_PACK="half", QBITS=8, GSIZE=-1, SYM=True, PACK_DIM="PackDim::MN", USE_FP=False, T_SCALE="half")
QCFG_W8A16_ASYM=QConfig(T_PACK="half", QBITS=8, GSIZE=-1, SYM=False, PACK_DIM="PackDim::MN", USE_FP=False, T_SCALE="half")
QCFG_W4A16_SYM=QConfig(T_PACK="half", QBITS=4, GSIZE=-1, SYM=True, PACK_DIM="PackDim::MN", USE_FP=False, T_SCALE="half")
QCFG_W4A16_ASYM=QConfig(T_PACK="half", QBITS=4, GSIZE=-1, SYM=False, PACK_DIM="PackDim::MN", USE_FP=False, T_SCALE="half")
QCFG_W2A16_SYM=QConfig(T_PACK="half", QBITS=2, GSIZE=-1, SYM=True, PACK_DIM="PackDim::MN", USE_FP=False, T_SCALE="half")
QCFG_W2A16_ASYM=QConfig(T_PACK="half", QBITS=2, GSIZE=-1, SYM=False, PACK_DIM="PackDim::MN", USE_FP=False, T_SCALE="half")

QCFG_W4A16_SYM_G128=QConfig(T_PACK="half", QBITS=4, GSIZE=128, SYM=True, PACK_DIM="PackDim::MN", USE_FP=False, T_SCALE="half")
QCFG_W4A16_ASYM_G128=QConfig(T_PACK="half", QBITS=4, GSIZE=128, SYM=False, PACK_DIM="PackDim::MN", USE_FP=False, T_SCALE="half")
QCFG_W2A16_SYM_G128=QConfig(T_PACK="half", QBITS=2, GSIZE=128, SYM=True, PACK_DIM="PackDim::MN", USE_FP=False, T_SCALE="half")
QCFG_W2A16_ASYM_G128=QConfig(T_PACK="half", QBITS=2, GSIZE=128, SYM=False, PACK_DIM="PackDim::MN", USE_FP=False, T_SCALE="half")

# for sm89
QCFG_W8A8_E4M3=QConfig(T_PACK="half", QBITS=8, GSIZE=-1, SYM=True, PACK_DIM="PackDim::K", USE_FP=True, T_SCALE="half")


@dataclasses.dataclass
class TileConfig:
    BM: int=64
    BN: int=64
    BK: int=64
    WM: int=2
    WN: int=2
    WK: int=1
    STAGE: int=2
    SPLITK: int=-1
    MMA: str="MMA_FP16_FP32"
    QCFGA: QConfig|NO_QUANT=dataclasses.field(default_factory=NO_QUANT)
    QCFGB: QConfig|NO_QUANT=dataclasses.field(default_factory=NO_QUANT)

    def to_str(self):
        return CFG_TEMPLATE.format(
            BM=self.BM, BN=self.BN, BK=self.BK,
            WM=self.WM, WN=self.WN, WK=self.WK,
            STAGE=self.STAGE, SPLITK=self.SPLITK, MMA=self.MMA,
            QCFGA=self.QCFGA.to_str(), QCFGB=self.QCFGB.to_str()
        )

    def __hash__(self) -> int:
        return hash(self.to_str())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TileConfig):
            return False
        return dataclasses.asdict(self) == dataclasses.asdict(other)
        

    @property
    def num_warps(self):
        return self.WM * self.WN * self.WK
    
    def warp_tile_mnk(self):
        return self.BM//self.WM, self.BN//self.WN, self.BK//self.WK
    
    def w_bits(self):
        return self.QCFGB.QBITS

    def a_bits(self):
        return self.QCFGA.QBITS
    
    def w_type(self) -> str:
        if isinstance(self.QCFGB, NO_QUANT):
            return MMA_TO_INP_TYPE[self.MMA]
        else:
            return self.QCFGB.qtype()

    def a_type(self) -> str:
        if isinstance(self.QCFGA, NO_QUANT):
            return MMA_TO_INP_TYPE[self.MMA]
        else:
            return self.QCFGB.qtype()

    def acc_type(self) -> str:
        acc_ty = MMA_TO_ACC_TYPE[self.MMA]
        return acc_ty

    def is_a_quant(self):
        return not isinstance(self.QCFGA, NO_QUANT)

    def is_b_quant(self):
        return not isinstance(self.QCFGB, NO_QUANT)
    
    def set_qcfg(self, qcfg: 'TileConfig') -> 'TileConfig':
        self.QCFGA = qcfg.QCFGA
        self.QCFGB = qcfg.QCFGB
        return self

    def smem_bytes_scale(self):
        smem_scale = 0
        # TODO: figure out when multistage is used 
        smem_scale += self.QCFGA.bytes_per_scale_zp() * self.BM * (1 if self.QCFGB.GSIZE == -1 else self.STAGE)
        smem_scale += self.QCFGB.bytes_per_scale_zp() * self.BN * (1 if self.QCFGB.GSIZE == -1 else self.STAGE)

        return smem_scale
    
    def smem_bytes_tile(self):
        acc_bytes = TYPE_TO_BITS[self.acc_type()] // 8

        smem_tile_a = self.STAGE * self.BM * self.BK * self.QCFGA.QBITS // 8
        smem_tile_b = self.STAGE * self.BN * self.BK * self.QCFGB.QBITS // 8

        # TODO: assume type of tile element is `half`
        return max(
            # 1. tile_a, tile_b
            smem_tile_b + smem_tile_a,
            # 2. tile_c (when slice-k is used)
            acc_bytes * (self.WK-1) * self.BM * self.BN,
        )

def get_info_from_qcfg_str(qcfg: str):
    splits = qcfg.split("_")
    wbits = int(splits[0].split("a")[0].split("w")[1])
    abits = int(splits[0].split("a")[1])
    gsize = int(splits[1].split("g")[1])
    sym = True if splits[2] == "sym" else False
    return wbits, abits, gsize, sym

def build_qcfgb_from_str(qcfg: str):
    splits = qcfg.split("_")
    wbits = int(splits[0].split("a")[0].split("w")[1])
    gsize = int(splits[1].split("g")[1])
    sym = True if splits[2] == "sym" else False

    ext_ty = splits[3] if len(splits) == 4 else ""

    return QConfig(T_PACK="half", QBITS=wbits, GSIZE=gsize, SYM=sym, PACK_DIM="PackDim::MN", USE_FP=ext_ty=="E4M3", T_SCALE="half")

QCFG_MAP = {
    **dict.fromkeys(
        ["fp16", "fp16_accfp16", "bf16"],
        TileConfig()
    ),

    # WxAx
    "w8a8_g-1_sym": TileConfig(QCFGA=QCFG_W8A8, QCFGB=QCFG_W8A8),
    "w8a8_g-1_sym_E4M3": TileConfig(QCFGA=QCFG_W8A8_E4M3, QCFGB=QCFG_W8A8_E4M3),
    "w4a4_g-1_sym": TileConfig(QCFGA=QCFG_W4A4, QCFGB=QCFG_W4A4),
    "w4a4_g128_sym": TileConfig(QCFGA=QCFG_W4A4_G128, QCFGB=QCFG_W4A4_G128),

    # WxA16
    **{
        k: TileConfig(QCFGB=build_qcfgb_from_str(k))
        for k in [f"w{wbits}a16_g{gsize}_{sym}{ty}"
                for wbits in [8,4,2]
                for gsize in [-1, 128]
                for sym in ["sym", "asym"]
                for ty in ["", "_accfp16", "_bf16"]]
    }
}


def get_possible_tile_list(arch: str, qcfg: str):
    arch_info = {
        "80": {
            "max_dyn_smem": "166912",
            "mma": {
                **dict.fromkeys(["MMA_FP16_FP32", "MMA_BF16_FP32"], (16, 8, 16)),
                "MMA_S8_K32": (16, 8, 32),
                "MMA_S4_K64": (16, 8, 64),
            }
        },
        "89": {
            "max_dyn_smem": "101376",
            "mma": {
                **dict.fromkeys(["MMA_FP16_FP32", "MMA_FP16_FP16", "MMA_BF16_FP32"], (16, 8, 16)),
                **dict.fromkeys(["MMA_S8_K32", "MMA_E4M3_K32", "MMA_E5M2_K32"], (16, 8, 32)),
                "MMA_S4_K64": (16, 8, 64),
            }
        },
    }

    def gen_possible_tile_list(arch: str, mma: str):
        mma_m, mma_n, mma_k = arch_info[arch][mma]
        pass



    # TODO: auto-generate this tile-list according to the hardware information
    tile_list = {
        "80": {
            "fp16": [
                TileConfig(BM=128, BN=256, BK=64, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=128, BN=128, BK=64, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=128, BN=128, BK=64, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_FP16_FP32") 
            ],

            "bf16": [
                TileConfig(BM=128, BN=256, BK=64, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_BF16_FP32"),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_BF16_FP32"),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_BF16_FP32"),
                TileConfig(BM=128, BN=128, BK=64, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_BF16_FP32"),
                TileConfig(BM=128, BN=128, BK=64, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_BF16_FP32") 
            ],

            "w8a8_g-1_sym": [
                TileConfig(BM=128, BN=256, BK=128,WM= 2,WN= 4,WK= 1,STAGE= 3,SPLITK= -1,MMA="MMA_S8_K32"),
                TileConfig(BM=128, BN=256, BK=64, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_S8_K32"),
                TileConfig(BM=128, BN=128, BK=128,WM= 2,WN= 2,WK= 1,STAGE= 4,SPLITK= -1,MMA="MMA_S8_K32"),
                TileConfig(BM=128, BN=128, BK=128,WM= 2,WN= 2,WK= 1,STAGE= 3,SPLITK= -1,MMA="MMA_S8_K32"),
                TileConfig(BM=128, BN=128, BK=64, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_S8_K32"),
                TileConfig(BM=128, BN=128, BK=64, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_S8_K32"),
                TileConfig(BM=128, BN=128, BK=64, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_S8_K32"),
                TileConfig(BM=128, BN=128, BK=64, WM=2, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_S8_K32"),
                TileConfig(BM=16, BN=128, BK=64, WM=1, WN=8, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_S8_K32"),
            ],
            "w4a4_g-1_sym": [
                TileConfig(BM=128, BN=128, BK=256, WM=2, WN=2, WK=1, STAGE=5, SPLITK=-1, MMA="MMA_S4_K64"),
                TileConfig(BM=128, BN=128, BK=256, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_S4_K64"),
                TileConfig(BM=128, BN=128, BK=256, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_S4_K64"),
                TileConfig(BM=128, BN=128, BK=128, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_S4_K64"),
                TileConfig(BM=128, BN=128, BK=128, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_S4_K64"),
                TileConfig(BM=128, BN=128, BK=128, WM=4, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_S4_K64"),
                TileConfig(BM=128, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_S4_K64"),
                TileConfig(BM=128, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_S4_K64"),
            ],

            **dict.fromkeys(["w4a16_g-1_sym", "w4a16_g128_sym", "w4a16_g-1_asym", "w4a16_g128_asym"],[
                TileConfig(BM=64, BN=256,BK= 32,WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=64, BN=256,BK= 32,WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=64, BN=128,BK= 32,WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=64, BN=128,BK= 64,WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=64, BN=128,BK=128,WM=1, WN=4, WK=2, STAGE=3, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=64, BN=64, BK=64, WM=1, WN=2, WK=2, STAGE=6, SPLITK=-1, MMA="MMA_FP16_FP32"),

                TileConfig(BM=48, BN=256,BK= 64, WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=48, BN=128,BK= 64, WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=48, BN=128,BK= 128,WM= 1,WN= 4,WK= 2,STAGE= 3,SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=48, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=4, SPLITK=-1, MMA="MMA_FP16_FP32"),

                TileConfig(BM=32, BN=256,BK= 64, WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=32, BN=128,BK= 64, WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=32, BN=128,BK= 128,WM= 1,WN= 4,WK= 2,STAGE=3, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=32, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=3, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=32, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=4, SPLITK=-1, MMA="MMA_FP16_FP32"),

                TileConfig(BM=16, BN=128,BK= 64, WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=16, BN=128,BK= 128,WM= 1,WN= 4,WK= 2,STAGE= 3,SPLITK= -1,MMA="MMA_FP16_FP32"),
                TileConfig(BM=16, BN=128,BK= 128,WM= 1,WN= 4,WK= 2,STAGE= 4,SPLITK= -1,MMA="MMA_FP16_FP32"),
                TileConfig(BM=16, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=3, SPLITK=-1, MMA="MMA_FP16_FP32"),
                TileConfig(BM=16, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=4, SPLITK=-1, MMA="MMA_FP16_FP32"),
            ]),
        },
        "89": {
            "fp16": [
                TileConfig(BM=128, BN=128, BK=64, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["fp16"]),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["fp16"]),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["fp16"]),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["fp16"]),
                TileConfig(BM=128, BN=128, BK=64, WM=4, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["fp16"]) 
            ],

            "fp16_accfp16": [
                TileConfig(BM=128, BN=128, BK=64, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["fp16_accfp16"]),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["fp16_accfp16"]),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["fp16_accfp16"]),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["fp16_accfp16"]),
                TileConfig(BM=128, BN=128, BK=64, WM=4, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["fp16_accfp16"]) 
            ],

            "bf16": [
                TileConfig(BM=128, BN=128, BK=64, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["bf16"]),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["bf16"]),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["bf16"]),
                TileConfig(BM=128, BN=128, BK=32, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["bf16"]),
                TileConfig(BM=128, BN=128, BK=64, WM=4, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["bf16"]) 
            ],

            "w8a8_g-1_sym": [
                TileConfig(BM=256, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                TileConfig(BM=256, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                TileConfig(BM=256, BN=128, BK=64,WM=2, WN=4, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                TileConfig(BM=256, BN=128, BK=64,WM=2, WN=4, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),

                TileConfig(BM=192, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                TileConfig(BM=192, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=64,WM=1, WN=4, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=64,WM=1, WN=4, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=64,WM=2, WN=4, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),

                # TileConfig(BM=128, BN=128, BK=128,WM=1, WN=8, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=128, BN=128, BK=128,WM=2, WN=2, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=128, BN=128, BK=128,WM=2, WN=4, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),

                # TileConfig(BM=128, BN=128, BK=64, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=128, BN=128, BK=64,WM= 4,WN= 2,WK= 1,STAGE= 4,SPLITK= -1, MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=128, BN=128, BK=64,WM= 2,WN= 4,WK= 1,STAGE= 4,SPLITK= -1, MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
            ],

            "w4a4_g-1_sym": [
                TileConfig(BM=256, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                TileConfig(BM=256, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                TileConfig(BM=256, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),

                TileConfig(BM=192, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                TileConfig(BM=192, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=128, WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),

                # TileConfig(BM=128, BN=128, BK=256, WM=1, WN=8, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=128, BN=128, BK=256, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=128, BN=128, BK=256, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),

                # TileConfig(BM=128, BN=128, BK=128, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=128, BN=128, BK=128, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=128, BN=128, BK=128, WM=4, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=128, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
            ],

            "w4a4_g128_sym": [
                # TileConfig(BM=256, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),

                # TileConfig(BM=192, BN=64, BK=128, WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=192, BN=64, BK=128, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                TileConfig(BM=192, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                TileConfig(BM=192, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                TileConfig(BM=192, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                TileConfig(BM=192, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),

                # TileConfig(BM=128, BN=64, BK=128, WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),

                # TileConfig(BM=128, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=5, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=128, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=128, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=128, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),

                # TileConfig(BM=128, BN=128, BK=128, WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
            ],

            **{k: [
                TileConfig(BM=64, BN=256,BK= 32,WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=64, BN=256,BK= 32,WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                # TileConfig(BM=64, BN=128,BK= 32,WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=64, BN=128,BK= 64,WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=64, BN=128,BK=128,WM=1, WN=4, WK=2, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                # TileConfig(BM=64, BN=64, BK=64, WM=1, WN=2, WK=2, STAGE=6, SPLITK=-1, MMA=QCFG_TO_MMA[k]),

                # TileConfig(BM=48, BN=256,BK= 64, WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                # TileConfig(BM=48, BN=128,BK= 64, WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                # TileConfig(BM=48, BN=128,BK= 128,WM= 1,WN= 4,WK= 2,STAGE= 3,SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                # TileConfig(BM=48, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),

                # TileConfig(BM=32, BN=256,BK= 64, WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                # TileConfig(BM=32, BN=128,BK= 64, WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                # TileConfig(BM=32, BN=128,BK= 128,WM= 1,WN= 4,WK= 2,STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                # TileConfig(BM=32, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                # TileConfig(BM=32, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),

                TileConfig(BM=16, BN=128,BK= 64, WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=128,BK= 128,WM= 1,WN= 4,WK= 2,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=128,BK= 128,WM= 1,WN= 4,WK= 2,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
            ] for k in [f"w4a16_g-1_{sym}{ext}" for sym in ["sym", "asym"] for ext in ["", "_accfp16", "_bf16"]]
            },

            **{k: [
                TileConfig(BM=64, BN=128,BK=128,WM=2, WN=2, WK=2, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=64, BN=128,BK=128,WM=2, WN=2, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]),

                TileConfig(BM=48, BN=256,BK= 128,WM= 1,WN= 4,WK= 1,STAGE= 3,SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=48, BN=256,BK= 128,WM= 1,WN= 4,WK= 2,STAGE= 3,SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=48, BN=128,BK= 128,WM= 1,WN= 4,WK= 2,STAGE= 3,SPLITK=-1, MMA=QCFG_TO_MMA[k]),

                TileConfig(BM=32, BN=128,BK= 128,WM= 1,WN= 4,WK= 2,STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=32, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=32, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),

                TileConfig(BM=16, BN=256,BK= 128, WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=128,BK= 128, WM=1, WN=2, WK=2, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=128,BK= 128,WM= 1,WN= 4,WK= 2,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=128,BK= 128,WM= 1,WN= 4,WK= 2,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=128,BK= 128,WM= 1,WN= 2,WK= 2,STAGE= 5,SPLITK= -1,MMA=QCFG_TO_MMA[k]),

                TileConfig(BM=16, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=64, BK=128, WM=1, WN=2, WK=2, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]),
            ] for k in [f"w4a16_g128_{sym}{ext}" for sym in ["sym", "asym"] for ext in ["", "_accfp16", "_bf16"]]
            },
            
            **{
                k: [
                TileConfig(BM=64, BN=256, BK=128,WM=1, WN= 4,WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=64, BN=128, BK=64, WM=2, WN=2, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]), 
                TileConfig(BM=32, BN=256, BK=64, WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA[k]), 
                TileConfig(BM=32, BN=256, BK=128,WM=1, WN= 4,WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),

                TileConfig(BM=16, BN=256, BK=64, WM=1, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA[k]), 
                TileConfig(BM=16, BN=256, BK=128,WM= 1,WN= 4,WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=256, BK=128,WM= 1,WN= 4,WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=256, BK=128,WM= 1,WN= 4,WK= 2,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=256, BK=128,WM= 1,WN= 4,WK= 2,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
            ] for k in [f"w2a16_g-1_{sym}{ext}" for sym in ["sym", "asym"] for ext in ["", "_accfp16", "_bf16"]]
            },

            **{
                k: [
                TileConfig(BM=64, BN=256, BK=128,WM=1, WN= 4,WK= 2,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=64, BN=256, BK=128,WM=2, WN= 4,WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=64, BN=256, BK=128,WM=1, WN= 4,WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=64, BN=128, BK=128,WM=2, WN= 2,WK= 2,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=64, BN=128, BK=128,WM=2, WN= 2,WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=64, BN=128, BK=128,WM=1, WN= 2,WK= 2,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA[k]),

                TileConfig(BM=48, BN=128, BK=128,WM=1, WN= 2,WK= 2,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=48, BN=128, BK=128,WM=1, WN= 2,WK= 2,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=48, BN=128, BK=128,WM=1, WN= 2,WK= 2,STAGE= 5,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=48, BN=256, BK=128,WM=1, WN= 4,WK= 2,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA[k]),

                TileConfig(BM=32, BN=128, BK=128,WM=1, WN= 2,WK= 2,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=32, BN=128, BK=128,WM=1, WN= 2,WK= 2,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=32, BN=128, BK=128,WM=1, WN= 2,WK= 2,STAGE= 5,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=32, BN=128, BK=128,WM=2, WN= 2,WK= 2,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=32, BN=128, BK=128,WM=2, WN= 2,WK= 2,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=32, BN=128, BK=128,WM=2, WN= 2,WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=32, BN=256, BK=128,WM=2, WN= 2,WK= 2,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),

                TileConfig(BM=16, BN=256, BK=128,WM= 1,WN= 4,WK= 2,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=256, BK=128,WM= 1,WN= 2,WK= 2,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=128, BK=128,WM= 1,WN= 2,WK= 2,STAGE= 5,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=128, BK=128,WM= 1,WN= 2,WK= 2,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
                TileConfig(BM=16, BN=128, BK=128,WM= 1,WN= 2,WK= 2,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA[k]),
            ] for k in [f"w2a16_g128_{sym}{ext}" for sym in ["sym", "asym"] for ext in ["", "_accfp16", "_bf16"]]
            },
        },
        "90": {
            "fp16": []
        }
    }

    return tile_list[arch][qcfg]
