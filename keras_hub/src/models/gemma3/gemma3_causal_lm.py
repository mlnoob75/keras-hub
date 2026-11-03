import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_causal_lm_preprocessor import (
    Gemma3CausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal

# Target number of tokens (across the batch) per chunk of a chunked prefill.
PREFILL_CHUNK_TOKENS = 2048

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.Gemma3CausalLM")
class Gemma3CausalLM(CausalLM):
    """An end-to-end multimodal Gemma3 model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    images and plain text inputs, or to autoregressively generate plain text
    similar to the data used for training. Note that the model is
    image-text in, text out.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_hub.samplers` objects to control the generation. By
    default, `"greedy"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

    Args:
        preprocessor: A `keras_hub.models.Gemma3CausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
        backbone: A `keras_hub.models.Gemma3Backbone` instance.
    """

    backbone_cls = Gemma3Backbone
    preprocessor_cls = Gemma3CausalLMPreprocessor

    def __init__(
        self,
        preprocessor,
        backbone,
        **kwargs,
    ):
        # === Layers ===
        self.preprocessor = preprocessor
        self.backbone = backbone

        # === Functional Model ===
        # This must be "backbone.input" i.e. the full input structure,
        # rather than "backbone.inputs" which is the flattened list of inputs.
        inputs = backbone.input
        hidden_state = backbone(inputs=inputs)
        outputs = backbone.token_embedding(hidden_state, reverse=True)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        sampler="greedy",
        **kwargs,
    ):
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            sampler=sampler,
            **kwargs,
        )

    def _normalize_generate_inputs(
        self,
        inputs,
    ):
        """Overrides the superclass' method to handle unbatched image inputs."""
        if tf and isinstance(inputs, tf.data.Dataset):
            return inputs.as_numpy_iterator(), False

        if self.preprocessor is None:
            return [inputs], False

        def normalize(x):
            if isinstance(x, str):
                return [x], True
            if tf and isinstance(x, tf.Tensor) and x.shape.rank == 0:
                return x[tf.newaxis], True
            return x, False

        if isinstance(inputs, dict):
            inputs["prompts"], input_is_scalar = normalize(inputs["prompts"])

            # If prompt is scalar, images can be either a 3D NumPy array/Tensor,
            # or, list of 3D NumPy arrays. Let's uprank images.
            if input_is_scalar and "images" in inputs:
                x = inputs["images"]
                if isinstance(x, np.ndarray) and len(x.shape) == 3:
                    inputs["images"] = [x]
                elif tf and isinstance(x, tf.Tensor) and x.shape.rank == 3:
                    inputs["images"] = x[tf.newaxis]
                elif isinstance(x, list):
                    inputs["images"] = [x]

            if "responses" in inputs:
                inputs["responses"], _ = normalize(inputs["responses"])
        else:
            inputs, input_is_scalar = normalize(inputs)

        return [inputs], input_is_scalar

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        img_embeddings=None,
        vision_mask=None,
        padding_mask=None,
        vision_indices=None,
        cache_update_mask=None,
        query_kernels=None,
    ):
        """Forward pass of `Gemma3CausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor with shape `(batch_size, max_length)`.
            cache: a tuple of per-layer caches, one entry per transformer
                layer. Each entry is a `(key, value)` tuple of dense float
                Tensors. `key` has shape
                `(batch_size, max_length, num_key_value_heads, head_dim)` and
                `value` has shape
                `(batch_size, num_key_value_heads, head_dim, max_length)`.
            cache_update_index: int, or int Tensor. The index of current inputs
                in the whole sequence.
            img_embeddings: a dense float Tensor with shape
                `(batch_size, image_sequence_length, hidden_dim)`.
            padding_mask: a dense int Tensor with shape
                `(batch_size, max_length)`.
            query_kernels: optional tuple with one entry per transformer
                layer, from `_query_kernels_for_generation()`.

        Returns:
            A (logits, hidden_states, cache) tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the updated decoding cache, in the same tuple-of-tuples format as
            the `cache` argument.
        """

        text_embeddings = self.backbone.token_embedding(token_ids)
        text_embeddings = text_embeddings * ops.cast(
            ops.sqrt(self.backbone.hidden_dim), text_embeddings.dtype
        )

        # Interleaving logic.
        ## == Interleaving text and images ==
        # Place the image embeddings in the right position in `text_embeddings`.
        if img_embeddings is not None:
            x = self.backbone.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=text_embeddings,
                vision_indices=vision_indices,
            )
        else:
            x = text_embeddings

        # Each decoder layer has a cache; we update them separately.
        # Cache is a tuple of per-layer (key, value) pairs — no stacking.
        caches = []
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[i]
            x, next_cache = transformer_layer(
                x,
                cache=current_cache,
                cache_update_index=cache_update_index,
                padding_mask=padding_mask,
                vision_mask=vision_mask,
                cache_update_mask=cache_update_mask,
                query_kernel=(
                    None if query_kernels is None else query_kernels[i]
                ),
            )
            caches.append(next_cache)
        cache = tuple(caches)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(
        self,
        token_ids,
        img_embeddings,
        vision_mask,
        padding_mask,
        vision_indices,
        query_kernels=None,
    ):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = ops.shape(token_ids)[0]
        max_length = (
            ops.shape(token_ids)[1]
            # + self.backbone.image_sequence_length
        )
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        # Allocate per-layer (key, value) caches as a tuple. Values are stored
        # sequence-last; see `CachedGemma3Attention.call`.
        key_shape = [batch_size, max_length, num_heads, head_dim]
        value_shape = [batch_size, num_heads, head_dim, max_length]
        cache = tuple(
            (
                ops.zeros(key_shape, dtype=self.compute_dtype),
                ops.zeros(value_shape, dtype=self.compute_dtype),
            )
            for _ in range(num_layers)
        )
        # Seed the cache.
        chunk_size = self._prefill_chunk_size(token_ids)
        if img_embeddings is None and chunk_size is not None:
            return self._prefill_in_chunks(
                token_ids, padding_mask, cache, chunk_size, query_kernels
            )
        logits, hidden_states, cache = self.call_with_cache(
            token_ids=token_ids,
            img_embeddings=img_embeddings,
            vision_mask=vision_mask,
            cache=cache,
            cache_update_index=0,
            padding_mask=padding_mask,
            vision_indices=vision_indices,
            cache_update_mask=None,
            query_kernels=query_kernels,
        )
        return hidden_states, cache

    def _prefill_chunk_size(self, token_ids):
        """Chunk length for `_prefill_in_chunks`, or `None` for one pass.

        Chunks hold about `PREFILL_CHUNK_TOKENS` tokens across the batch, so
        each chunk is still a large matmul. Small batches and short sequences
        fit in one chunk and keep the single-pass prefill.
        """
        batch_size, max_length = token_ids.shape
        if batch_size is None or max_length is None:
            return None
        chunk_size = max(64, PREFILL_CHUNK_TOKENS // batch_size)
        return chunk_size if chunk_size < max_length else None

    def _prefill_in_chunks(
        self, token_ids, padding_mask, cache, chunk_size, query_kernels=None
    ):
        """Seed the cache in fixed-size chunks, up to the longest prompt.

        Inputs are padded to the full generation length, so a single-pass
        prefill also runs the model over every position that generation will
        later fill. Here a loop with a data-dependent trip count stops at the
        end of the longest prompt.

        Prompts are right-padded, so the causal mask alone keeps every prompt
        token from attending to padding. Positions past a prompt get
        placeholder cache entries, which generation overwrites before they are
        attended to.
        """
        batch_size, max_length = token_ids.shape
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        num_chunks = (ops.max(row_lengths) + chunk_size - 1) // chunk_size
        hidden_states = ops.zeros(
            (batch_size, max_length, self.backbone.hidden_dim),
            dtype=self.compute_dtype,
        )

        def cond(i, hidden_states, cache):
            return i < num_chunks

        def body(i, hidden_states, cache):
            # Keep the last chunk inside the sequence. It may then recompute
            # some positions, which writes the same cache values again.
            start = ops.minimum(i * chunk_size, max_length - chunk_size)
            chunk = ops.slice(token_ids, [0, start], [batch_size, chunk_size])
            _, chunk_hidden_states, cache = self.call_with_cache(
                token_ids=chunk,
                cache=cache,
                cache_update_index=start,
                query_kernels=query_kernels,
            )
            hidden_states = ops.slice_update(
                hidden_states, [0, start, 0], chunk_hidden_states
            )
            return i + 1, hidden_states, cache

        i = ops.convert_to_tensor(0, "int32")
        _, hidden_states, cache = ops.while_loop(
            cond, body, (i, hidden_states, cache)
        )
        return hidden_states, cache

    def _query_kernels_for_generation(self):
        """Query kernels laid out for the query matmul, one per layer.

        The stored layout needs a transpose before the matmul. Done inside
        the prefill and decoding loops, it is repeated at every iteration
        (XLA does not hoist transposes of loop-invariant weights), so it is
        done once here instead.
        """
        return tuple(
            layer.attention.transposed_query_kernel()
            for layer in self.backbone.transformer_layers
        )

    def generate_step(self, inputs, stop_token_ids=[106]):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"token_ids"` and `"padding_mask"`.

        Args:
            inputs: A dictionary with two keys `"token_ids"` and
                `"padding_mask"` and batched tensor values.
            stop_token_ids: Tuple of id's of end token's to stop on. If all
                sequences have produced a new stop token, generation
                will stop.
        """

        token_ids, padding_mask, images, vision_mask, vision_indices = (
            inputs["token_ids"],
            inputs["padding_mask"],
            inputs.get("images", None),
            inputs.get("vision_mask", None),
            inputs.get("vision_indices", None),
        )

        # Determine if we have actual images to process.
        # After preprocessing, images shape is (batch, num_images, h, w, 3).
        # For text-only input, num_images=0 (static shape).
        # We use static shape check which returns a Python int, not a tensor.
        num_images = 0
        if (
            images is not None
            and hasattr(images, "shape")
            and len(images.shape) > 1
        ):
            num_images = images.shape[
                1
            ]  # Static shape, returns Python int or None

        if not self.backbone.text_only_model and num_images:
            # Handle an unbatched image. Unlike `token_ids` and
            # `padding_mask`, this will not automatically be upranked.
            if len(ops.shape(images)) == 4:
                images = ops.expand_dims(images, axis=0)
            if len(ops.shape(vision_mask)) == 1:
                vision_mask = ops.expand_dims(vision_mask, axis=0)
            if len(ops.shape(vision_indices)) == 1:
                vision_indices = ops.expand_dims(vision_indices, axis=0)
            img_embeddings = self.backbone.vision_encoder(images)
        else:
            img_embeddings = None
            vision_mask = None
            vision_indices = None

        query_kernels = self._query_kernels_for_generation()

        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids,
            img_embeddings,
            vision_mask,
            padding_mask,
            vision_indices,
            query_kernels,
        )

        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, index - 1], [batch_size, 1])
            sliced_cache_update_mask = ops.slice(
                ~padding_mask, [0, index - 1], [batch_size, 1]
            )
            logits, hidden_states, cache = self.call_with_cache(
                token_ids=prompt,
                cache=cache,
                cache_update_index=cache_update_index,
                cache_update_mask=sliced_cache_update_mask,
                query_kernels=query_kernels,
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
                cache,
            )

        token_ids = self.sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )

        # Compute an output padding mask with the token ids we updated.
        if stop_token_ids is not None:
            # Build a mask of `stop_token_ids` locations not in the original
            # prompt (not in locations where `padding_mask` is True).
            end_locations = any_equal(
                token_ids, stop_token_ids, ops.logical_not(padding_mask)
            )

            end_locations = ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after end_locations.
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            # Without early stopping, all locations will have been updated.
            padding_mask = ops.ones_like(token_ids, dtype="bool")
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "images": images,
        }

    def generate(
        self,
        inputs,
        max_length=None,
        stop_token_ids="auto",
        strip_prompt=False,
    ):
        # If `auto`, add `<end_of_turn>` as a stop token too.
        if self.preprocessor is None and stop_token_ids == "auto":
            raise ValueError(
                "A `preprocessor` must be attached to the model if "
                '`stop_token_ids="auto"`. Currently `preprocessor=None`. To '
                "call `generate()` with preprocessing detached, either pass "
                "`stop_token_ids=None` to always generate until `max_length` "
                "or pass a tuple of token ids that should terminate generation "
                "as `stop_token_ids`."
            )
        elif stop_token_ids == "auto":
            stop_token_ids = [
                self.preprocessor.tokenizer.end_token_id,
                self.preprocessor.tokenizer.token_to_id("<end_of_turn>"),
            ]

        return super().generate(
            inputs,
            max_length=max_length,
            stop_token_ids=stop_token_ids,
            strip_prompt=strip_prompt,
        )
