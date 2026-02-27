from dataclasses import dataclass
import torch
import torch.nn as NN
import torch.nn.utils.parametrize as parametrize
from src.utils_quantization import FakeQuantParametrization 
from fvcore.nn import FlopCountAnalysis

@dataclass
class LayerStat:
    name: str
    idx: int
    input_channels: int
    output_channels: int
    output_prunning_ratio: float
    weight_bitrate: int
    MACs_uncompressed: int
    activation_bitrate: int
    kernel_height: int = 1
    kernel_width: int = 1
    feature_map_height: int = 1
    feature_map_width: int = 1

@dataclass
class LayerCompressionStat:
    name: str
    idx: int
    BOBs_compression_rate: float 
    BOBs_compressed: int
    BOBs_uncompressed: int
    MACs_compressed: int
    MACs_uncompressed: int

@dataclass
class ComparisonResults:
    total_BOBs_compressed: int
    total_MACs_compressed: int
    total_BOBs_uncompressed: int
    total_MACs_uncompressed: int
    total_bobs_compression_rate: float
    layer_results: list[LayerCompressionStat]

    def print(self):
        print(f"{'='*55}")
        print(f"  Compression Summary")
        print(f"{'='*55}")
        print(f"  BOBs compressed:    {self.total_BOBs_compressed:>15,}")
        print(f"  BOBs uncompressed:  {self.total_BOBs_uncompressed:>15,}")
        print(f"  MACs compressed:    {self.total_MACs_compressed:>15,}")
        print(f"  MACs uncompressed:  {self.total_MACs_uncompressed:>15,}")
        print(f"  BOBs compression:   {self.total_bobs_compression_rate:>14.1%}")
        print(f"{'='*55}")
        print(f"  {'Layer':<10} {'BOBs rate':>10} {'MACs comp':>12} {'MACs uncomp':>12}")
        print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
        for layer in self.layer_results:
            print(
                f"  {layer.name:<10} "
                f"{layer.BOBs_compression_rate:>10.1%} "
                f"{layer.MACs_compressed:>12,} "
                f"{layer.MACs_uncompressed:>12,}"
            )
        print(f"{'='*55}")

@torch.no_grad()
def compare_model(compressed_model : NN.Module, default_weight_bitrate: int = 8, default_activation_bitrate: int = 8):
    layer_stats = []
    for idx, (name, module) in enumerate(compressed_model.named_modules()):
        if isinstance(module, NN.Linear):
            weights = module.weight
            prunning_ratio, output_ch, input_ch = output_prunning_ratio_linear(weights=weights)
            weight_bitrate = default_weight_bitrate
            if parametrize.is_parametrized(module, "weight"):
                plist = module.parametrizations["weight"]
                for p in plist:
                    if isinstance(p, FakeQuantParametrization):
                        weight_bitrate = int(p.quantizer.get_bitwidth().item())

            stats = LayerStat(
                name=name,
                idx=idx,
                input_channels=input_ch,
                output_channels=output_ch,
                output_prunning_ratio=prunning_ratio,
                weight_bitrate=weight_bitrate,
                MACs_uncompressed=input_ch*output_ch,
                activation_bitrate=default_activation_bitrate,
            )
            layer_stats.append(stats)

    return calculate_compression_stats(layer_stats=layer_stats, default_weight_bitrate=default_weight_bitrate, default_activation_bitrate=default_activation_bitrate)
    
def output_prunning_ratio_linear(weights: torch.Tensor):
    output_ch,input_ch = weights.shape
    num_zero_rows = (weights.abs().sum(dim=1) == 0).sum().item()
    prunning_ratio =  num_zero_rows / output_ch

    return prunning_ratio, output_ch, input_ch
    
def calculate_compression_stats(layer_stats : list[LayerStat], default_weight_bitrate: int = 8, default_activation_bitrate: int = 8):
    total_BOBs_compressed = 0
    total_MACs_compressed = 0
    total_BOBs_uncompressed = 0
    total_MACs_uncompressed = 0
    layer_compression_stats = []

    for idx, layer in enumerate(layer_stats):
        if idx == 0:
            prev_output_prunning_ratio = 0
        else:
            prev_output_prunning_ratio = layer_stats[idx-1].output_prunning_ratio

        MACs_compressed = (1- prev_output_prunning_ratio) \
        * layer.input_channels \
        * (1 - layer.output_prunning_ratio) \
        * layer.output_channels \
        * layer.feature_map_width \
        * layer.feature_map_height \
        * layer.kernel_height \
        * layer.kernel_width

        BOBs_compressed = MACs_compressed * layer.activation_bitrate * layer.weight_bitrate
        BOBs_uncompressed = layer.MACs_uncompressed * default_activation_bitrate * default_weight_bitrate

        total_BOBs_compressed += BOBs_compressed 
        total_MACs_compressed += MACs_compressed

        total_MACs_uncompressed +=layer.MACs_uncompressed 
        total_BOBs_uncompressed += BOBs_uncompressed

        BOBs_compression_rate =  BOBs_compressed / BOBs_uncompressed

        stat = LayerCompressionStat(
            name = layer.name,
            idx=layer.idx, 
            BOBs_compression_rate = BOBs_compression_rate,
            BOBs_compressed=int(BOBs_compressed),
            BOBs_uncompressed=BOBs_uncompressed,
            MACs_compressed=int(MACs_compressed),
            MACs_uncompressed=int(layer.MACs_uncompressed)
        )

        layer_compression_stats.append(stat)
    
    total_BOBs_compression_rate = total_BOBs_compressed / total_BOBs_uncompressed 

    return ComparisonResults(
        total_BOBs_compressed=int(total_BOBs_compressed),
        total_BOBs_uncompressed=total_BOBs_uncompressed,
        total_MACs_compressed=int(total_MACs_compressed),
        total_MACs_uncompressed=total_MACs_uncompressed,
        total_bobs_compression_rate=total_BOBs_compression_rate,
        layer_results=layer_compression_stats
    )




