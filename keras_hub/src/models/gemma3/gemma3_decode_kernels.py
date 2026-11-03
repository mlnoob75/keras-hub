"""Pallas GPU kernel for the Gemma3 attention block during decoding.

While decoding one token at a time, XLA runs the attention block as a chain
of about ten small kernels (query/key normalization, rotary embedding, two
cache updates, two matmuls and three softmax passes), and on a GPU each of
them costs a few microseconds of launch and drain time. This kernel does the
whole block in one pass per batch row and key/value head: it normalizes and
rotates the new query and key, writes the new key and value into the cache in
place, and attends with an online softmax over the live range of cached keys
only.

JAX on GPU only. Callers check `can_use_decode_attention_kernel()` first.
"""

import functools

# Keys per block in the kernel's loop over the cache.
_BLOCK_KEYS = 32
# Query rows per key/value head are loaded as a block of this many rows,
# the smallest tensor-core dot.
_BLOCK_ROWS = 16


def can_use_decode_attention_kernel(
    num_query_heads, num_key_value_heads, head_dim, dtype
):
    """Whether `decode_attention_block()` supports these sizes and dtype."""
    import jax

    if dtype not in ("bfloat16", "float16"):
        return False
    if num_query_heads // num_key_value_heads > _BLOCK_ROWS:
        return False
    # Triton blocks need power-of-two sizes; rotary halves need >= 16.
    if head_dim < 32 or head_dim & (head_dim - 1):
        return False
    capability = getattr(jax.devices()[0], "compute_capability", None)
    # Pre-Ampere GPUs lack the shared memory for the pipelined loads.
    return capability is not None and float(capability) >= 8.0


def _rms_norm(x1, x2, scale1, scale2, dim, epsilon):
    """`RMSNormalization` of a vector split in two halves, in float32."""
    import jax
    import jax.numpy as jnp

    sum_sq = jnp.sum(x1 * x1, axis=-1, keepdims=True)
    sum_sq += jnp.sum(x2 * x2, axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(sum_sq / dim + epsilon)
    return x1 * inv * scale1, x2 * inv * scale2


def _decode_attention_block_kernel(
    bounds_ref,
    update_ref,
    query_ref,
    key_ref,
    value_ref,
    query_scale_ref,
    key_scale_ref,
    cos_ref,
    sin_ref,
    key_cache_ref,
    value_cache_ref,
    out_ref,
    new_key_cache_ref,
    new_value_cache_ref,
    *,
    groups,
    seq_len,
    query_normalization,
    epsilon,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    f32 = jnp.float32
    dtype = query_ref.dtype
    batch = pl.program_id(0)
    kv_head = pl.program_id(1)
    start = bounds_ref[0]
    index = bounds_ref[1]
    update = update_ref[batch] != 0
    head_dim = cos_ref.shape[0]
    half = head_dim // 2
    lo, hi = pl.ds(0, half), pl.ds(half, half)

    def halves(ref):
        return ref[lo].astype(f32), ref[hi].astype(f32)

    cos1, cos2 = halves(cos_ref)
    sin1, sin2 = halves(sin_ref)
    query_scale1, query_scale2 = halves(query_scale_ref)
    key_scale1, key_scale2 = halves(key_scale_ref)

    def rotate(x1, x2):
        # Rotary embedding (`x * cos + rotate_half(x) * sin`) on halves,
        # from inputs rounded to the activation dtype.
        x1 = x1.astype(dtype).astype(f32)
        x2 = x2.astype(dtype).astype(f32)
        return x1 * cos1 - x2 * sin1, x2 * cos2 + x1 * sin2

    # The query rows of this key/value head, as a block of `_BLOCK_ROWS`.
    rows = pl.ds(kv_head * groups, _BLOCK_ROWS)
    row_mask = (jnp.arange(_BLOCK_ROWS) < groups)[:, None]
    query1 = plgpu.load(query_ref.at[batch, rows, lo], mask=row_mask, other=0)
    query2 = plgpu.load(query_ref.at[batch, rows, hi], mask=row_mask, other=0)
    query1, query2 = _rms_norm(
        query1.astype(f32),
        query2.astype(f32),
        query_scale1[None, :],
        query_scale2[None, :],
        head_dim,
        epsilon,
    )
    query1, query2 = rotate(query1, query2)
    query1 = (query1.astype(dtype).astype(f32) * query_normalization).astype(
        dtype
    )
    query2 = (query2.astype(dtype).astype(f32) * query_normalization).astype(
        dtype
    )

    key1, key2 = _rms_norm(
        key_ref[batch, kv_head, lo].astype(f32)[None, :],
        key_ref[batch, kv_head, hi].astype(f32)[None, :],
        key_scale1[None, :],
        key_scale2[None, :],
        head_dim,
        epsilon,
    )
    key1, key2 = rotate(key1[0], key2[0])
    key1 = key1.astype(dtype)
    key2 = key2.astype(dtype)
    value = value_ref[batch, kv_head, :]

    def body(block, carry):
        row_max, row_sum, acc = carry
        offset = block * _BLOCK_KEYS
        positions = offset + jnp.arange(_BLOCK_KEYS)
        keys = pl.ds(offset, _BLOCK_KEYS)
        in_cache = positions < seq_len
        cached_key1 = plgpu.load(
            key_cache_ref.at[batch, keys, kv_head, lo],
            mask=in_cache[:, None],
            other=0,
        )
        cached_key2 = plgpu.load(
            key_cache_ref.at[batch, keys, kv_head, hi],
            mask=in_cache[:, None],
            other=0,
        )
        cached_value = plgpu.load(
            value_cache_ref.at[batch, kv_head, :, keys],
            mask=in_cache[None, :],
            other=0,
        )
        # The new key and value are only written to the cache at the end.
        new = (positions == index) & update
        cached_key1 = jnp.where(new[:, None], key1[None, :], cached_key1)
        cached_key2 = jnp.where(new[:, None], key2[None, :], cached_key2)
        cached_value = jnp.where(new[None, :], value[:, None], cached_value)

        logits = jnp.dot(query1, cached_key1.T, preferred_element_type=f32)
        logits += jnp.dot(query2, cached_key2.T, preferred_element_type=f32)
        # Round like the einsum path, whose logits are in the input dtype.
        logits = logits.astype(dtype).astype(f32)
        attended = (positions >= start) & (positions <= index)
        logits = jnp.where(attended[None, :], logits, -jnp.inf)
        new_max = jnp.maximum(row_max, jnp.max(logits, axis=1))
        rescale = jnp.exp(row_max - new_max)
        probs = jnp.exp(logits - new_max[:, None])
        row_sum = row_sum * rescale + jnp.sum(probs, axis=1)
        acc = acc * rescale[:, None] + jnp.dot(
            probs.astype(dtype), cached_value.T, preferred_element_type=f32
        )
        return new_max, row_sum, acc

    init = (
        jnp.full((_BLOCK_ROWS,), -jnp.inf, f32),
        jnp.zeros((_BLOCK_ROWS,), f32),
        jnp.zeros((_BLOCK_ROWS, head_dim), f32),
    )
    _, row_sum, acc = jax.lax.fori_loop(
        start // _BLOCK_KEYS, index // _BLOCK_KEYS + 1, body, init
    )
    out = (acc / row_sum[:, None]).astype(out_ref.dtype)
    plgpu.store(out_ref.at[batch, rows, :], out, mask=row_mask)

    # Rows whose current position is part of the prompt keep their cache.
    plgpu.store(
        new_key_cache_ref.at[batch, index, kv_head, lo],
        key1,
        mask=jnp.full((half,), update),
    )
    plgpu.store(
        new_key_cache_ref.at[batch, index, kv_head, hi],
        key2,
        mask=jnp.full((half,), update),
    )
    plgpu.store(
        new_value_cache_ref.at[batch, kv_head, :, index],
        value,
        mask=jnp.full((head_dim,), update),
    )


def decode_attention_block(
    query,
    key,
    value,
    query_scale,
    key_scale,
    cos,
    sin,
    key_cache,
    value_cache,
    start,
    index,
    update,
    query_normalization,
    epsilon,
):
    """Single-token attention block with an in-place cache update.

    Args:
        query: `(batch, num_query_heads, head_dim)` projected query.
        key: `(batch, num_key_value_heads, head_dim)` projected key.
        value: `(batch, num_key_value_heads, head_dim)` projected value.
        query_scale: `(head_dim,)` float32 query norm multiplier
            (`1 + scale`).
        key_scale: `(head_dim,)` float32 key norm multiplier.
        cos: `(head_dim,)` rotary embedding cosines at `index`.
        sin: `(head_dim,)` rotary embedding sines at `index`.
        key_cache: `(batch, seq, num_key_value_heads, head_dim)`.
        value_cache: `(batch, num_key_value_heads, head_dim, seq)`.
        start: int scalar, first attended cache position.
        index: int scalar, position of the new token.
        update: `(batch,)` bool, whether to write each row's new key and
            value into the cache.
        query_normalization: float, query scaling before the dot product.
        epsilon: float, epsilon of the query and key norms.

    Returns:
        A tuple of the `(batch, num_query_heads, head_dim)` attention output
        and the updated key and value caches.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    batch_size, num_heads, head_dim = query.shape
    seq_len, num_kv_heads = key_cache.shape[1], key_cache.shape[2]
    bounds = jnp.stack([start, index]).astype("int32")
    kernel = functools.partial(
        _decode_attention_block_kernel,
        groups=num_heads // num_kv_heads,
        seq_len=seq_len,
        query_normalization=query_normalization,
        epsilon=epsilon,
    )
    return pl.pallas_call(
        kernel,
        grid=(batch_size, num_kv_heads),
        out_shape=[
            jax.ShapeDtypeStruct(query.shape, query.dtype),
            jax.ShapeDtypeStruct(key_cache.shape, key_cache.dtype),
            jax.ShapeDtypeStruct(value_cache.shape, value_cache.dtype),
        ],
        input_output_aliases={9: 1, 10: 2},
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=3),
    )(
        bounds,
        update.astype("int32"),
        query,
        key,
        value,
        query_scale.astype("float32"),
        key_scale.astype("float32"),
        cos,
        sin,
        key_cache,
        value_cache,
    )
