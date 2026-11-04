"""ComfyUI-OmniSampler -- a combined diffusion sampler.

Adds four sampler names to the stock KSampler / KSamplerSelect dropdowns
(omni_hybrid, omni_hybrid_sde, omni_hybrid_cfg_pp, omni_hybrid_gap) plus three
nodes: a simple SAMPLER-object node (one named setting), an advanced one (every
knob), and the full KSampler with the A/B preset switch.
"""

from .omni_nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    register_samplers,
)

register_samplers()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

