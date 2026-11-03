from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.samplers.sampler import Sampler


@keras_hub_export("keras_hub.samplers.GreedySampler")
class GreedySampler(Sampler):
    """Greedy sampler class.

    This sampler is implemented on greedy search, i.e., always picking up the
    token of the largest probability as the next token.

    Examples:
    ```python
    causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Pass by name to compile.
    causal_lm.compile(sampler="greedy")
    causal_lm.generate(["Keras is a"])

    # Pass by object to compile.
    sampler = keras_hub.samplers.GreedySampler()
    causal_lm.compile(sampler=sampler)
    causal_lm.generate(["Keras is a"])
    ```
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def get_next_token(self, probabilities):
        return ops.argmax(probabilities, axis=-1)

    def _get_next_token_from_logits(self, logits):
        # The softmax is monotonic, so its argmax is the argmax of the logits:
        # skip computing the distribution over the whole vocabulary. Only
        # when neither step is overridden and the temperature is positive.
        overridden = (
            type(self).compute_probabilities
            is not Sampler.compute_probabilities
            or type(self).get_next_token is not GreedySampler.get_next_token
        )
        if overridden or not self.temperature > 0:
            return super()._get_next_token_from_logits(logits)
        return ops.argmax(logits, axis=-1)
