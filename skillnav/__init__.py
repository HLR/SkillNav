"""SkillNav: Mixture of Skill-Based Vision-and-Language Navigation Agents.

The actual training and inference code lives under ``skillnav.backbones``:

    skillnav.backbones.scalevln  -> SkillNav on ScaleVLN (ViT-B/16)
    skillnav.backbones.srdf      -> SkillNav on VLN-SRDF (InternViT-6B)

Each backbone keeps the upstream baseline's directory layout
(``maps_nav_src/`` and ``map_nav_src/`` respectively) so that internal
imports like ``from utils.x import y`` and ``from moe.z import w`` keep
resolving when the corresponding directory is the current working dir.
Use the bash scripts under each backbone's ``scripts/`` folder.
"""

__version__ = "0.1.0"
