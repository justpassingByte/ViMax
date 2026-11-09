import logging
import os
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from interfaces import CharacterInScene, ImageOutput
from langchain_core.messages import HumanMessage, SystemMessage



prompt_template_front = \
"""Generate a full-body, front-view portrait of character {identifier} based on the following description, with a pure solid white background. The character should be centered in the frame from head to toe, gazing straight ahead, standing naturally on two feet with empty hands and arms relaxed at sides. Natural expression.
SOLO CHARACTER ONLY: Render ONLY this single character with NO other characters, NO pets, NO held objects, NO food, NO floating weapons, NO lightning/smoke effects, and NO text or labels.
Character Features & Outfit: {features}
Visual Style: {style}
"""

prompt_template_side = \
"""Generate a full-body, side-view portrait of character {identifier} with a pure solid white background, facing left, standing naturally on two feet with empty hands and arms relaxed at sides.
CRITICAL CONSISTENCY: Must wear the EXACT same clothing, outfit colors, hairstyle, eyeglasses, and physical appearance as the front view. Render ONLY this single character with NO props, NO pets, NO floating weapons, NO effects, and NO text.
Character Features & Outfit: {features}
Visual Style: {style}
"""

prompt_template_back = \
"""Generate a full-body, back-view portrait of character {identifier} seen from behind, with a pure solid white background, centered, empty hands and arms relaxed at sides. No facial features visible.
CRITICAL CONSISTENCY: Must wear the EXACT same clothing, outfit colors, and hairstyle as the front view, seen strictly from behind. Front-only accessories (such as a necktie, bowtie, front shirt placket, buttons, or chest badge) hang down the chest in front and are HIDDEN from behind — keep the upper back of a shirt/coat completely smooth and NEVER draw a necktie hanging down the back. Render ONLY this single character with NO props, NO pets, NO floating weapons, NO effects, and NO text.
Character Features & Outfit: {features}
Visual Style: {style}
"""

prompt_template_quadruped_front = \
"""Generate a full-body, front-view natural four-legged animal reference portrait of {identifier} based on the following description, on a pure solid white background. The animal stands naturally on all four paws on flat ground facing straight ahead toward the camera.
CRITICAL ANATOMY: Pure natural non-anthropomorphic quadruped animal (all four paws on ground, NEVER bipedal, NEVER standing on two hind legs, NO human arms or hands, NO clothing). SOLO SUBJECT ONLY: NO humans, NO held objects, NO food, NO lightning effects, NO text or labels.
Animal Features: {features}
Visual Style: {style}
"""

prompt_template_quadruped_side = \
"""Generate a full-body, left-facing side profile turnaround reference portrait of {identifier} on a pure solid white background, standing naturally on all four paws on flat ground facing left, clearly showing head, round body, all four legs, and tail.
CRITICAL CONSISTENCY: Must keep the EXACT same fur colors, tabby stripe pattern, body shape, ear shape, and eye color as the front view. Pure natural four-legged quadruped animal (NEVER bipedal, NO human hands or clothes, NO props, NO lightning effects, NO text).
Animal Features: {features}
Visual Style: {style}
"""

prompt_template_quadruped_back = \
"""Generate a full-body, rear back-view turnaround reference portrait of {identifier} on a pure solid white background, standing naturally on all four paws seen cleanly from behind showing back fur markings, ears, and tail.
CRITICAL CONSISTENCY: Must keep the EXACT same fur colors, stripe pattern, and proportions as the front view. Pure natural four-legged quadruped animal (NEVER bipedal, NO clothes, NO props, NO text).
Animal Features: {features}
Visual Style: {style}
"""


_QUADRUPED_MARKERS = (
    "quadruped",
    "four-legged",
    "all four paws",
    "bốn chân",
    "mèo mướp",
    "con mèo",
    "chú mèo",
    "bé mèo",
    "dạng mèo",
    "mèo xám",
    "mèo hoang",
    "mèo béo",
    "mèo cam",
    "tabby cat",
    "fat cat",
    "chubby cat",
    "house cat",
    "gray cat",
    "grey cat",
    "stray cat",
    "experimental cat",
    "dog",
    "puppy",
    "wolf",
    "fox",
    "horse",
    "tiger",
    "lion",
)

_HUMANOID_MARKERS = (
    "dạng người",
    "hình dạng người",
    "human form",
    "humanoid",
    "người đàn ông",
    "nam sinh",
    "cô gái",
    "thiếu niên",
    "phụ nữ",
    "tiến sĩ",
    "nhà khoa học",
    "dị nhân",
    "trench coat",
    "school uniform",
    "hooded",
)


def _is_quadruped_character(character: CharacterInScene) -> bool:
    combined = " ".join([
        character.identifier_in_scene or "",
        character.static_features or "",
        character.dynamic_features or "",
    ]).lower()
    if any(marker in combined for marker in _HUMANOID_MARKERS):
        return False
    return any(marker in combined for marker in _QUADRUPED_MARKERS)


class CharacterPortraitsGenerator:
    def __init__(
        self,
        image_generator,
    ):
        self.image_generator = image_generator


    async def generate_front_portrait(
        self,
        character: CharacterInScene,
        style: str,
        reference_image_paths: Optional[List[str]] = None,
    ) -> ImageOutput:
        features = f"(static) {character.static_features or ''}; (dynamic) {character.dynamic_features or ''}"
        template = prompt_template_quadruped_front if _is_quadruped_character(character) else prompt_template_front
        style_val = style or "Modern cinematic, crisp photorealistic details"
        prompt = template.format(
            identifier=character.identifier_in_scene,
            features=features,
            style=style_val,
        )
        refs = [str(p).strip() for p in (reference_image_paths or []) if str(p).strip()][:4]
        if refs:
            prompt = (
                f"{prompt.rstrip()}\n"
                f"HYBRID COMIC-TO-ANIME REDRAW: Strictly match the character's exact facial structure, hair color/style, eye color, fur/tabby markings, and signature outfit from the attached reference image(s), while redrawing cleanly in `{style_val}` on a pure solid white background. "
                "REMOVE all comic panel borders, speech bubbles, text letters, SFX, screentones, and extra background figures."
            )
            image_output = await self.image_generator.generate_single_image(
                prompt=prompt,
                reference_image_paths=refs,
            )
        else:
            image_output = await self.image_generator.generate_single_image(
                prompt=prompt,
            )
        return image_output

    async def generate_side_portrait(
        self,
        character: CharacterInScene,
        front_image_path: str,
        style: str = "",
    ) -> ImageOutput:
        features = f"(static) {character.static_features or ''}; (dynamic) {character.dynamic_features or ''}"
        template = prompt_template_quadruped_side if _is_quadruped_character(character) else prompt_template_side
        prompt = template.format(
            identifier=character.identifier_in_scene,
            features=features,
            style=style or "Modern cinematic, crisp photorealistic details",
        )
        image_output = await self.image_generator.generate_single_image(
            prompt=prompt,
            reference_image_paths=[front_image_path],
        )
        return image_output


    async def generate_back_portrait(
        self,
        character: CharacterInScene,
        front_image_path: str,
        style: str = "",
    ) -> ImageOutput:
        features = f"(static) {character.static_features or ''}; (dynamic) {character.dynamic_features or ''}"
        template = prompt_template_quadruped_back if _is_quadruped_character(character) else prompt_template_back
        prompt = template.format(
            identifier=character.identifier_in_scene,
            features=features,
            style=style or "Modern cinematic, crisp photorealistic details",
        )
        image_output = await self.image_generator.generate_single_image(
            prompt=prompt,
            reference_image_paths=[front_image_path],
        )
        return image_output
