"""vLLM plugin registering the DSV4-0731 vision wrapper.

Loaded via the `vllm.general_plugins` entry point, which vLLM invokes in every
worker process before the model is built.
"""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV4VisionForCausalLM",
        "dsv4_vision_vllm.model:DeepseekV4VisionForCausalLM",
    )
