"""ComfyUI-OmniSampler -- a combined diffusion sampler.

Adds four sampler names to the stock KSampler / KSamplerSelect dropdowns
(omni_hybrid, omni_hybrid_sde, omni_hybrid_cfg_pp,
omni_hybrid_cfgpp_author_pref) plus two nodes that expose the same sampler
knobs: a full KSampler with an A/B preset switch, and a SAMPLER-object node for
SamplerCustom / SamplerCustomAdvanced.
"""

from .omni_nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    register_samplers,
)

register_samplers()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

