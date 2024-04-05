from .var import VAR
from .vqvae import VQVAE


def build_var(
    vae: VQVAE, depth: int,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    aln=1, aln_gamma_init=1e-3, shared_aln=False, layer_scale=-1,
    tau=4, cos_attn=False,
    flash_if_available=True, fused_if_available=True,
):
    return VAR(
        vae_local=vae, patch_nums=patch_nums,
        depth=depth, embed_dim=depth*64, num_heads=depth, drop_path_rate=0.1 * depth/24,
        aln=aln, aln_gamma_init=aln_gamma_init, shared_aln=shared_aln, layer_scale=layer_scale,
        tau=tau, cos_attn=cos_attn,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    )
