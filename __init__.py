"""ComfyUI-OmniSampler -- a combined diffusion sampler.

Adds three sampler names to the stock KSampler / KSamplerSelect dropdowns
(omni_hybrid, omni_hybrid_sde, omni_hybrid_cfg_pp) plus a node that exposes
every knob.
"""

from .omni_nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    register_samplers,
)

register_samplers()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

