import json
import random
import dataclasses

@dataclasses.dataclass
class QLinearConfig:
    w_bits: int = 16
    w_gsize: int = -1
    w_sym: bool = False
    w_clip: tuple[float,float] = (1.0, 1.0)
    a_bits: int = 16
    a_gsize: int = -1
    a_sym: bool = True
    a_clip: tuple[float,float] = (1.0, 1.0)

    def __str__(self):
        return f"W{self.w_bits}A{self.a_bits}_g{self.w_gsize}_{'sym' if self.w_sym else 'asym'}"
    
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> 'QLinearConfig':
        return QLinearConfig(
            w_bits=d["w_bits"],
            w_gsize=d["w_gsize"],
            w_sym=d["w_sym"],
            w_clip=d["w_clip"],
            a_bits=d["a_bits"],
            a_gsize=d["a_gsize"],
            a_sym=d["a_sym"],
            a_clip=d["a_clip"]
        )


@dataclasses.dataclass
class QExpertConfig:
    gate: QLinearConfig
    up: QLinearConfig
    down: QLinearConfig

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
    
    def qmap(self) -> dict[str, QLinearConfig]:
        return {
            "gate": self.gate,
            "up": self.up,
            "down": self.down
        }

@dataclasses.dataclass
class QLayerConfig:
    experts: dict[str, QExpertConfig]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

@dataclasses.dataclass
class QModelConfig:
    layers: dict[str, QLayerConfig]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

def build_qmodel_cfg_from_json(json_path: str) -> QModelConfig:
    with open(json_path, "r") as f:
        qmodel_cfg: dict[str, dict[str, dict[str, dict]]] = json.load(f)

    LT = qmodel_cfg.pop("LT", None)
    layers = {}
    for layer_idx, layer_cfg in qmodel_cfg.items():
        experts = {}
        for expert_idx, expert_cfg in layer_cfg["experts"].items():
            gate_cfg = QLinearConfig(**expert_cfg["gate"])
            up_cfg = QLinearConfig(**expert_cfg["up"])
            down_cfg = QLinearConfig(**expert_cfg["down"])
            experts[expert_idx] = QExpertConfig(gate=gate_cfg, up=up_cfg, down=down_cfg)
        layers[layer_idx] = QLayerConfig(experts=experts)
    return QModelConfig(layers=layers)


def build_uni_qexpert_cfg(qcfg: QLinearConfig) -> QExpertConfig:
    return QExpertConfig(
        gate=qcfg,
        up=qcfg,
        down=qcfg
    )

def build_uni_qlayer_cfg(qcfg: QLinearConfig, num_experts: int) -> QLayerConfig:
    return QLayerConfig(
        experts={str(i): build_uni_qexpert_cfg(qcfg) for i in range(num_experts)}
    )

def build_uni_qmodel_cfg(qcfg: QLinearConfig, num_layers: int, num_experts: int) -> QModelConfig:
    return QModelConfig(
        layers={str(i): build_uni_qlayer_cfg(qcfg, num_experts) for i in range(num_layers)}
    )

def get_all_wbits(qmodel_cfg: QModelConfig) -> list[tuple[int,int, bool]]:
    bits = set()
    for layer in qmodel_cfg.layers.values():
        for expert in layer.experts.values():
            for qcfg in expert.qmap().values():
                bits.add((qcfg.w_bits, qcfg.w_gsize, qcfg.w_sym))
    return list(bits)


if __name__ == "__main__":
    qmodel_cfg = QModelConfig(layers={
        0: QLayerConfig(experts={
            0: QExpertConfig(
                gate=QLinearConfig(w_bits=8, w_gsize=-1, w_sym=True, a_bits=8, a_gsize=-1, a_sym=True),
                up=QLinearConfig(w_bits=8, w_gsize=-1, w_sym=True, a_bits=8, a_gsize=-1, a_sym=True),
                down=QLinearConfig(w_bits=8, w_gsize=-1, w_sym=True, a_bits=8, a_gsize=-1, a_sym=True)
            ),
            1: QExpertConfig(
                gate=QLinearConfig(w_bits=8, w_gsize=-1, w_sym=True, a_bits=8, a_gsize=-1, a_sym=True),
                up=QLinearConfig(w_bits=8, w_gsize=-1, w_sym=True, a_bits=8, a_gsize=-1, a_sym=True),
                down=QLinearConfig(w_bits=8, w_gsize=-1, w_sym=True, a_bits=8, a_gsize=-1, a_sym=True)
            )
        }),
        1: QLayerConfig(experts={
            0: QExpertConfig(
                gate=QLinearConfig(w_bits=4, w_gsize=-1, w_sym=True, a_bits=8, a_gsize=-1, a_sym=True),
                up=QLinearConfig(w_bits=4, w_gsize=-1, w_sym=True, a_bits=8, a_gsize=-1, a_sym=True),
                down=QLinearConfig(w_bits=4, w_gsize=-1, w_sym=True, a_bits=8, a_gsize=-1, a_sym=True)
            ),
            1: QExpertConfig(
                gate=QLinearConfig(w_bits=8, w_gsize=-1, w_sym=True, a_bits=8, a_gsize=-1, a_sym=True),
                up=QLinearConfig(w_bits=8, w_gsize=-1, w_sym=True, a_bits=8, a_gsize=-1, a_sym=True),
                down=QLinearConfig(w_bits=8, w_gsize=-1, w_sym=True, a_bits=8, a_gsize=-1, a_sym=True)
            )
        })
    })
    json_data = json.dumps(dataclasses.asdict(qmodel_cfg))
    import jsbeautifier as jsb
    json_data = jsb.beautify(json_data)
    # print(json_data)

    q = QModelConfig(json.loads(json_data))
    # print(q)
    # # build_qmodel_cfg_from_json("./qconfigs/ds2.json")
    x = build_qmodel_cfg_from_json(f"./qconfigs/qwen2_moe_GPTQ.json")
    print(x.layers["0"].experts["0"])
