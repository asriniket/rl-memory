"""Video encoder for Multi-Scale Embodied Memory (MEM).

Extends the SigLIP ViT used by Pi0 with space-time separable attention as
described in "MEM: Multi-Scale Embodied Memory for Vision Language Action
Models" (Torne et al. 2025). Every 4-th ViT layer runs a causal-temporal
attention over same-patch tokens across frames *before* the spatial
attention, reusing the layer's own Q/K/V/output projection weights and
LayerNorm (no new learnable parameters for the attention mechanism itself).
A sinusoidal temporal position embedding with boundary condition `e(0) = 0`
is added to the temporal branch's input at those layers.

Space-time attention follows the "Divided Space-Time" formulation of
TimeSformer (Bertasius et al. 2021), which the paper cites. At a temporal
layer the residual update is

    z  = x + α_time[LN(x + e(t)), causal_mask]   (temporal residual)
    x' = z + α_space[LN(z)]                       (spatial residual)

where α_time and α_space share the same Q/K/V/out projections and the same
LayerNorm. At non-temporal layers this reduces to the base SigLIP block
(pure spatial attention), and at `num_frames == 1` the temporal branch is
disabled entirely — the module is then numerically identical to the base
ViT.

To permit end-to-end fine-tuning on small downstream datasets without
catastrophic drift of the vision tower, the attention's Q/K/V/output
projections and the MLP's two dense layers carry optional LoRA adapters
(Hu et al. 2021). The sidecar parameters are named `lora_a` / `lora_b`
alongside the original `kernel` / `bias`, so when the training freeze
filter pins the base weights and keeps `.*lora.*` trainable, the encoder
learns a low-rank delta while the pretrained SigLIP manifold is kept
intact.

The module is parameter-compatible with `siglip._Module` (same parameter
tree structure and names for the base weights), so Pi0.5 checkpoints load
without modification. Only the attention *pattern*, the temporal position
embedding, and the LoRA sidecars are new.
"""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
from openpi.models.siglip import decode_variant
from openpi.models.siglip import get_posemb
import openpi.training.sharding as sharding

# Every 4-th layer (1-indexed) applies temporal attention, i.e. the 4th, 8th,
# 12th, ... layer. In 0-indexed form this is `(layer_idx + 1) % 4 == 0`.
_TEMPORAL_LAYER_STRIDE = 4


def _temporal_posemb(num_frames: int, embedding_dim: int, dtype=jnp.float32):
    """Sinusoidal temporal position embedding with e(0) = 0.

    Frames are ordered from oldest (index 0) to current (index `num_frames - 1`).
    The current frame is assigned `t = 0`, so its embedding is zero and the
    K = 1 case exactly matches the non-temporal ViT initialization.
    """
    pos = (jnp.arange(num_frames) - (num_frames - 1)).astype(jnp.float32)
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    min_period, max_period = 1.0, 1.0e4
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2.0 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    pe = jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)
    pe = pe - pe[num_frames - 1 : num_frames]
    return pe.astype(dtype)


class _LoRADense(nn.Module):
    """`nn.Dense`-compatible layer with optional LoRA sidecar.

    Stores the base weight/bias under the same parameter names as `nn.Dense`
    (`kernel`, `bias`) so pretrained ViT checkpoints load unchanged. When
    `lora_config` is set, `lora_a` (in, rank) and `lora_b` (rank, out)
    contribute an additive low-rank update `x @ A @ B * alpha/rank`.
    """

    features: int
    dtype_mm: str = "float32"
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    bias_init: nn.initializers.Initializer = nn.initializers.normal(stddev=1e-6)
    lora_config: lora.LoRAConfig | None = None

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        dtype = x.dtype
        kernel = self.param("kernel", self.kernel_init, (in_features, self.features))
        bias = self.param("bias", self.bias_init, (self.features,))
        y = jnp.dot(x, kernel.astype(dtype)) + bias.astype(dtype)
        if (cfg := self.lora_config) is not None:
            lora_a = self.param("lora_a", cfg.init_fn, (in_features, cfg.rank))
            lora_b = self.param("lora_b", cfg.init_fn, (cfg.rank, self.features))
            delta = jnp.dot(jnp.dot(x, lora_a.astype(dtype)), lora_b.astype(dtype))
            y = y + delta * cfg.scaling_value
        return y


class _LoRADenseGeneral(nn.Module):
    """`nn.DenseGeneral`-compatible layer with optional LoRA sidecar.

    Preserves the `kernel`/`bias` parameter shapes and names used by
    `flax.linen.DenseGeneral`, so the query/key/value/out projections inside
    `_LoRAMultiHeadDotProductAttention` match what pretrained SigLIP ViT
    checkpoints expect. LoRA factors are shape-matched to the base kernel.
    """

    features: int | Sequence[int]
    axis: int | Sequence[int] = -1
    use_bias: bool = True
    dtype_mm: str = "float32"
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    lora_config: lora.LoRAConfig | None = None

    @nn.compact
    def __call__(self, x):
        features_t = (self.features,) if isinstance(self.features, int) else tuple(self.features)
        axis_t = (self.axis,) if isinstance(self.axis, int) else tuple(self.axis)
        axis_t = tuple(a if a >= 0 else x.ndim + a for a in axis_t)
        in_dims = tuple(x.shape[a] for a in axis_t)

        dtype = x.dtype
        kernel = self.param("kernel", self.kernel_init, in_dims + features_t)
        contract_axes = tuple(range(len(in_dims)))
        y = jnp.tensordot(x, kernel.astype(dtype), axes=(axis_t, contract_axes))
        if self.use_bias:
            bias = self.param("bias", self.bias_init, features_t)
            y = y + bias.astype(dtype)

        if (cfg := self.lora_config) is not None:
            lora_a = self.param("lora_a", cfg.init_fn, in_dims + (cfg.rank,))
            lora_b = self.param("lora_b", cfg.init_fn, (cfg.rank,) + features_t)
            delta = jnp.tensordot(x, lora_a.astype(dtype), axes=(axis_t, contract_axes))
            delta = jnp.tensordot(delta, lora_b.astype(dtype), axes=((-1,), (0,)))
            y = y + delta * cfg.scaling_value
        return y


class _LoRAMultiHeadDotProductAttention(nn.Module):
    """Multi-head attention with optional LoRA on the Q/K/V/out projections.

    The four projection submodules are named `query`, `key`, `value`, and
    `out`, with internal `kernel`/`bias` parameters matching
    `flax.linen.MultiHeadDotProductAttention`, so pretrained SigLIP base
    weights load unchanged. When `lora_config` is set, each projection gains
    `lora_a` and `lora_b` parameters that train a low-rank delta.
    """

    num_heads: int
    deterministic: bool = True
    dtype_mm: str = "float32"
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    lora_config: lora.LoRAConfig | None = None

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, mask=None):
        features = inputs_q.shape[-1]
        assert features % self.num_heads == 0, "num_heads must divide the feature dim"
        head_dim = features // self.num_heads
        dtype = inputs_q.dtype

        def _proj(x, name):
            return _LoRADenseGeneral(
                features=(self.num_heads, head_dim),
                axis=-1,
                dtype_mm=self.dtype_mm,
                kernel_init=self.kernel_init,
                lora_config=self.lora_config,
                name=name,
            )(x)

        q = _proj(inputs_q, "query")
        k = _proj(inputs_kv, "key")
        v = _proj(inputs_kv, "value")

        q = q / jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32)).astype(dtype)
        attn_weights = jnp.einsum("...qhd,...khd->...hqk", q, k)
        if mask is not None:
            big_neg = jnp.finfo(attn_weights.dtype).min
            attn_weights = jnp.where(mask, attn_weights, big_neg)
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(dtype)
        attn_out = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v)

        return _LoRADenseGeneral(
            features=features,
            axis=(-2, -1),
            dtype_mm=self.dtype_mm,
            kernel_init=self.kernel_init,
            lora_config=self.lora_config,
            name="out",
        )(attn_out)


class _LoRAMlpBlock(nn.Module):
    """ViT MLP block with optional LoRA on both dense layers.

    Submodules are named `Dense_0` and `Dense_1`, matching the auto-naming
    that `siglip.MlpBlock` produces, so pretrained ViT weights load
    unchanged.
    """

    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"
    lora_config: lora.LoRAConfig | None = None

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        d = x.shape[-1]
        hidden = self.mlp_dim or 4 * d
        x = _LoRADense(
            features=hidden,
            dtype_mm=self.dtype_mm,
            lora_config=self.lora_config,
            name="Dense_0",
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return _LoRADense(
            features=d,
            dtype_mm=self.dtype_mm,
            lora_config=self.lora_config,
            name="Dense_1",
        )(x)


class VideoEncoder1DBlock(nn.Module):
    """ViT encoder block with optional causal-temporal attention (TimeSformer-style).

    At a temporal layer (every `_TEMPORAL_LAYER_STRIDE`-th layer, 1-indexed),
    temporal attention runs first — over same-patch tokens across frames with
    a causal mask — and its output is added back to `x` via a residual.
    Spatial attention then runs on the result. The two attentions share the
    same Q/K/V/output projections and the same pre-attention LayerNorm (no
    new learnable parameters for the attention mechanism itself). At
    non-temporal layers this reduces to the base SigLIP block.

    If `lora_config` is provided, the attention's Q/K/V/out projections and
    the MLP's two dense layers carry LoRA adapters.
    """

    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    num_frames: int = 1
    lora_config: lora.LoRAConfig | None = None

    @nn.compact
    def __call__(self, x, layer_idx, deterministic=True):  # noqa: FBT002
        k = self.num_frames
        embed_dim = x.shape[-1]
        x = sharding.activation_sharding_constraint(x)

        ln_sa = nn.LayerNorm(dtype=self.dtype_mm)
        attn = _LoRAMultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype_mm=self.dtype_mm,
            lora_config=self.lora_config,
            name="MultiHeadDotProductAttention_0",
        )

        if k > 1:
            is_temporal = ((layer_idx + 1) % _TEMPORAL_LAYER_STRIDE == 0).astype(self.dtype_mm)
            temporal_pe = _temporal_posemb(k, embed_dim, dtype=self.dtype_mm)
            bk, n, _ = x.shape
            b = bk // k
            # Additive PE, gated so non-temporal layers see zero shift.
            pe_add = (is_temporal * temporal_pe).reshape(1, k, 1, embed_dim)
            z_hat = (x.reshape(b, k, n, embed_dim) + pe_add).reshape(bk, n, embed_dim)

            # Temporal attention: attend across frames at each spatial position.
            y_t = ln_sa(z_hat)
            y_t = y_t.reshape(b, k, n, embed_dim).transpose(0, 2, 1, 3).reshape(b * n, k, embed_dim)
            causal = jnp.tril(jnp.ones((k, k), dtype=jnp.bool_))[None, None, :, :]
            y_t = attn(y_t, y_t, mask=causal)
            y_t = y_t.reshape(b, n, k, embed_dim).transpose(0, 2, 1, 3).reshape(bk, n, embed_dim)
            # Temporal residual, gated so non-temporal layers pass through unchanged.
            x = x + is_temporal * y_t
            x = sharding.activation_sharding_constraint(x)

        # Spatial attention (identical to base ViT on non-temporal layers).
        y = ln_sa(x)
        y = attn(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = _LoRAMlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
            lora_config=self.lora_config,
            name="MlpBlock_0",
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, {}


class VideoEncoder(nn.Module):
    """Scanned stack of `VideoEncoder1DBlock` layers.

    Produces the same parameter tree as `siglip.Encoder` (with `scan=True`),
    so pre-trained SigLIP weights load without remapping.
    """

    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    num_frames: int = 1
    lora_config: lora.LoRAConfig | None = None

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        block = nn.remat(
            VideoEncoder1DBlock,
            prevent_cse=False,
            static_argnums=(3,),
            policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
        )
        x, _ = nn.scan(
            block,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast),
            length=self.depth,
        )(
            name="encoderblock",
            dtype_mm=self.dtype_mm,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            num_frames=self.num_frames,
            lora_config=self.lora_config,
        )(x, jnp.arange(self.depth), deterministic)
        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), {}


class _VideoModule(nn.Module):
    """Video-encoding counterpart to `siglip._Module` with `pool_type='none'`.

    Input: `(b, k, h, w, c)`. Output: `(b, n, num_classes)` projected tokens for
    the current (last) frame, so the downstream VLA sees the same number of
    tokens as in the single-image case.
    """

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None
    num_heads: int = 12
    posemb: str = "learn"
    dropout: float = 0.0
    head_zeroinit: bool = True
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    num_frames: int = 1
    lora_config: lora.LoRAConfig | None = None

    @nn.compact
    def __call__(self, video, *, train=False):
        if video.ndim != 5:
            raise ValueError(f"VideoModule expects (b, k, h, w, c) input, got shape {video.shape}")
        b, k, h_in, w_in, c_in = video.shape
        if k != self.num_frames:
            raise ValueError(f"Expected num_frames={self.num_frames}, got video frame count {k}")

        video = jnp.asarray(video, jnp.float32)
        frames = video.reshape(b * k, h_in, w_in, c_in)

        x = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(frames)

        bk, h, w, d = x.shape
        x = x.reshape(bk, h * w, d)
        x = x + get_posemb(self, self.posemb, (h, w), d, "pos_embedding", jnp.float32)
        x = nn.Dropout(rate=self.dropout)(x, not train)
        x = x.astype(self.dtype_mm)

        x, _ = VideoEncoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            num_frames=self.num_frames,
            lora_config=self.lora_config,
            name="Transformer",
        )(x, deterministic=not train)

        tokens = x.reshape(b, k, h * w, d)
        current = tokens[:, k - 1]

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            current = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)(current)
        return current, {}


def Module(  # noqa: N802
    num_classes=None,
    *,
    variant=None,
    num_frames: int = 1,
    lora_config: lora.LoRAConfig | None = None,
    **kw,
):
    return _VideoModule(
        num_classes,
        num_frames=num_frames,
        lora_config=lora_config,
        **{**decode_variant(variant), **kw},
    )
