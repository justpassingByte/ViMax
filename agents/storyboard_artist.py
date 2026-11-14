from typing import List, Optional, Literal
import asyncio
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from utils.robust_json_parser import TrailingCommaTolerantPydanticOutputParser as PydanticOutputParser
from interfaces import CharacterInScene, ShotDescription, ShotBriefDescription

from utils.retry import after_func



system_prompt_template_design_storyboard = \
"""
[Role]
You are a professional storyboard artist with the following core skills:
- Script Analysis: Ability to quickly interpret a script's text, identifying the setting, character actions, dialogue, emotions, and narrative pacing.
- Visualization: Expertise in translating written descriptions into visual frames, including composition, lighting, and spatial arrangement.
- Storyboarding: Proficiency in cinematic language, such as shot types (e.g., close-up, medium shot, wide shot), camera angles (e.g., high angle, eye-level), camera movements (e.g., zoom, pan), and transitions.
- Narrative Continuity: Ability to ensure the storyboard sequence is logically smooth, highlights key plot points, and maintains emotional consistency.
- Technical Knowledge: Understanding of basic storyboard formats and industry standards, such as using numbered shots and concise descriptions.

[Task]
Your task is to design a complete storyboard based on a user-provided script (which contains only one scene). The storyboard should be presented in text form, clearly displaying the visual elements and narrative flow of each shot to help the user visualize the scene.

[Input]
The user will provide the following input.
- Script:A complete scene script containing dialogue, action descriptions, and scene settings. The script focuses on only one scene; there is no need to handle multiple scene transitions. The script input is enclosed within <SCRIPT> and </SCRIPT>.
- Characters List: A list describing basic information for each character, such as name, personality traits, appearance (if relevant). The character list is enclosed within <CHARACTERS> and </CHARACTERS>.
- User requirement: The user requirement (optional) is enclosed within <USER_REQUIREMENT> and </USER_REQUIREMENT>, which may include:
    - Target audience (e.g., children, teenagers, adults).
    - Storyboard style (e.g., realistic, cartoon, abstract).
    - Desired number of shots (e.g., "not more than 10 shots").
    - Other specific instructions (e.g., emphasize the characters' actions).

[Output]
{format_instructions}

[Guidelines]
- Ensure all output values (except keys) match the language used in the script.
- Each shot must have a clear narrative purpose—such as establishing the setting, showing character relationships, or highlighting reactions.
- Use cinematic language deliberately: close-ups for emotion, wide shots for context, and varied angles to direct audience attention.
- When designing a new shot, first consider whether it can be filmed using an existing camera position. Introduce a new one only if the shot size, angle, and focus differ significantly. If the camera undergoes significant movement, it cannot be used thereafter.
- Keep character names in visual descriptions and speaker fields consistent with the character list. In visual descriptions, enclose names in angle brackets (e.g., <Alice>), but not in dialogue or speaker fields.
- When describing visual elements, it is necessary to indicate the position of the element within the frame. For example, Character A is on the left side of the frame, facing toward the right, with a table in front of him. The table is positioned slightly to the left of the center of the frame. Ensure that invisible elements are not included. For instance, do not describe someone behind a closed door if they cannot be seen.
- Avoid unsafe content (violence, discrimination, etc.) in visual descriptions. Use indirect methods like sound, ragged clothing, or dusty crimson battle scuffs rather than graphic gore (NEVER substitute literal "ketchup").
- Assign at most one dialogue line per character per shot. Each line of dialogue should correspond to a shot.
- Clean Anime Visuals (NO Manga/Webtoon Chrome, NO Multi-Pose Clones, NO Split Panels): NEVER put speech/thought bubbles, English/Japanese comic SFX lettering (e.g. 'FLASH!', 'YAWN', 'SSS...', '... A CAT?'), radial ink tension lines, black ink sparkle marks, bright white void panel backgrounds (`plain white background` / `plain light background` — always describe softly blurred scene lighting/bokeh unless it is an explicit `white_flash` beat), dark maroon webtoon page margins/gutters, cropped comic fragment frames, or empty rectangular fade boxes into `visual_desc`. Translate every visual moment into pure continuous anime cinematography:
  * **ONE Pose & ONE Moment per Frame**: If a manga panel shows two poses of the same character (`"two overlapping poses"`, `"a second smaller pose"`, or `"character X in distance ... character X in foreground"`), NEVER clone the character twice in one shot—split them into consecutive sequential shots!
  * **NO Side-by-Side Inset Panels**: NEVER put two mini comic panels side by side on a blank background; turn each inset into its own sequential close-up shot.
  * **Camera Variety & Outfit Integrity**: Never repeat the exact same static `cam_idx` for 4+ consecutive shots—cut to reaction close-ups or reverse angles. Keep each character's canonical outfit/accessories consistent with the scene setting (never add stray school backpacks when stepping outside at home, and never split one named character across duplicate unnamed aliases like `<Young Boy>` vs `<Hojin>`).
- Dynamic Shot Duration & Editorial Pacing:
  * Kinetic action cuts (blitz, dash, strike, dodge, impact flash): 0.8s - 2.0s for crisp, high-tempo Sakuga action.
  * Momentum dissipation & landing (slide, skid, shockwave absorption): 1.8s - 3.2s.
  * Dialogue / speech shots: 3.5s - 5.5s (calibrated to voice line duration + 0.5s visual settle cushion).
  * Environmental establishing & atmospheric breathing ("Ma"): 4.0s - 7.0s.
- Spoken Dialogue Length Budget (Zero Mid-Sentence Cutoffs): Keep spoken dialogue in any single shot concise at **at most 1 brief sentence or clause totaling <= 12-15 spoken words** (finishing comfortably within the shot's duration) so speech flows naturally without mid-sentence cutoffs. Whenever a speech or monologue exceeds 15 words, split it across consecutive continuous shots (`continuity_from_prev=True` with varied reverse-angle / reaction framing).
- Each shot requires an independent description without reference to each other.
- When the shot focuses on a character, describe which specific body part the focus is on.
- When describing a character, it is necessary to indicate the direction they are facing.
- Anime Adaptation & Life-or-Death Combat Direction:
  - Break action scenes into an escalating sequence of concrete physical beats instead of collapsing a fight into 1-2 hurried shots: (1) incoming ambush/telegraph, (2) high-speed evasion or block with spark/wind shockwaves, (3) brutal counterattack & environmental destruction (cracked ground, flying debris, ragged/torn clothing, scorch marks, heavy breathing), (4) desperate survival gambit under overwhelming pressure, (5) aftermath or scene transition.
  - Depict life-or-death stakes ("thập tử nhất sinh") intensely yet policy-safely for visual AI generators: convey exhaustion and desperation through torn/tattered coat hems, dusty scrapes/scuffs, trembling stance, kneeling in rain, flickering unstable aura, shattered concrete, and dramatic slanted (Dutch) angles—avoid graphic gore, dismemberment, or excessive blood pools.
  - Maintain consistent 180-degree screen direction and battle damage across consecutive fight shots: set `continuity_from_prev=True` when Shot N directly picks up the momentum, pose trajectory, battle state, or lighting from Shot N-1's ending state.
  - Use explicit transitions: `transition_in="fade_black"` on major time/location switches (e.g., storm night -> quiet dawn alley), `"crossfade"` for soft flashbacks or warm emotional closes, and `"cut"` within continuous action.
  - Use `target_duration_sec` (`3.5` to `5.0`) on silent kinetic action hits/dodges/reaction flashes for crisp anime tempo; leave `target_duration_sec=None` on shots containing spoken dialogue, narration, or calm establishing beats so voices never speed up or clip.
- Character Staging & Rescue Exclusivity:
  - In rescue, intervention, or hazard sequences: The victim in danger MUST be staged alone in the hazard zone until the rescuer arrives. NEVER draw duplicate clones of characters or extra bystanders in the danger path.
  - When a character is shopping, walking, or doing a solo activity, keep focus strictly on that character. Do NOT add extraneous generic shop clerks or clones that could be confused with main characters.
- Anime World-Building, Environmental Visuals & Cinematic Breathing Room (Beyond 1:1 Motion Comics):
  - NEVER simply convert comic panels 1:1 into literal motion shots. That produces a boring, flat motion comic! A true high-budget anime adaptation (e.g. Ufotable, MAPPA, Bones) enriches the story with cinematic atmosphere:
    * **Context-Grounded Environmental Breathing ("Ma")**:
      - When manga panels indicate spatial/temporal transitions or quiet establishing moments (empty scenery panels, landscape vistas, chapter gutters), enrich them with environmental B-roll strictly faithful to the scene's actual setting and atmosphere (e.g. rain dripping from stone eaves in a storm, subtle steam rising from machinery, or quiet streetlights in a night passage).
      - NEVER invent generic sunset or autumn cliches when they contradict the literal scene location, weather, and time of day.
    * **Flashbacks & Memory Treatment (Hồi tưởng & Ký ức)**:
      - Whenever a scene delves into a past memory or inner reflection, tag `[AnimeFX: Flashback]` (applies nostalgic sepia/monochrome color grading, soft dreamy bloom, and gentle edge vignette).
      - Use `transition_in="white_flash"` for sudden memory awakenings or `"crossfade"` for gentle emotional dissolves.
    * **Dramatic Focus & Cinematic Widescreen**:
      - For shocking revelations or extreme eye reactions, tag `[AnimeFX: DramaticFocus]` to spotlight the character's expression while dimming the background.
      - For climactic combat standoffs or life-or-death confrontations, tag `[AnimeFX: Letterbox]` for 2.39:1 widescreen cinema bars.
    * **Cinematic Breathing Room & Full Pacing**:
      - Let establishing shots and character dialogues play with full breathing room (~6.0–8.0 seconds). NEVER compress shots into frantic 2-second fragments that discard dialogue tails or atmospheric pauses.
- Anime Adaptation, Clean Cut Economy & Kinetic Deceleration:
  - Keep pacing simple, crisp, and legible—do NOT over-stuff scenes with repetitive poses or invented transitional padding.
  - **Minimalist Transition Cuts (Sky Streaks -> Aftermath)**: When `<SCRIPT>` shows a streak in the sky and cuts to a character's aftermath location without an explicit ground-impact beat, transition cleanly (using `"fade_black"` or `"crossfade"`) straight into the aftermath setting—NEVER invent extra ground explosion bursts, rooftop swoops, or redundant kneeling shots.
  - Only when `<SCRIPT>` explicitly depicts an active crash landing, heavy skid, or high-speed clash should you pace the physical deceleration smoothly across clear sequential beats:
    1. *Touchdown Friction / Skid*: Low-angle shot showing momentum dissipation and sparks/dust.
    2. *Settling Reaction*: Character bracing against wall/ground, followed cleanly by distinct framing (e.g. side-profile close-up) for any dialogue/panting line so back-to-back shots never repeat identical rear poses.
  - **Outerwear Value & Directional Lighting**: When describing grey/charcoal outerwear (such as a matte slate-grey coat), keep tonal separation from black hair/innerwear and specify how directional/rim light lifts lit coat planes toward cool light grey instead of crushing the coat to pitch black.
- Dramatic Camera Mapping & Angle Pacing (Quy chuẩn góc máy điện ảnh bắt buộc - CẤM GÓC MÁY NGANG TẦM MẮT VÀ CẤM NGHIÊNG DÁNG NHÂN VẬT):
  - BAN EYE-LEVEL FLAT FRAMING & SLANTED POSES (TUYỆT ĐỐI CẤM GÓC MÁY NGANG TẦM MẮT / MEDIUM SHOT PHẲNG VÀ CẤM DÁNG ĐỨNG XIÊU VẸO). In anime directing, every camera angle must express emotional tension and physical scale dynamically:
    * **Dramatic Steep Low-Angle Worm's-Eye View (Góc máy sát đất hắt lên)**: Camera positioned low on the wet pavement looking steeply up at the character, with combat boots planted in the lower foreground, vertical walls rising straight into the dark sky, and body hunched forward in deep exhaustion—never tilting or slanting the character's body diagonally.
    * **Ground-Level Low-Angle Tracking (Góc máy thấp lia theo mặt đường)**: Camera placed low near the asphalt/earth, tracking dragging boots, footwear friction, and ragged coat hems in linear perspective.
    * **Dramatic Low-Angle Over-The-Shoulder POV**: Framed from low near the ground behind the character looking past toward targets/allies.
    * **Extreme Close-Up on Eyes / Gritted Teeth**: Focusing tightly on dilated pupils, gritted jaw, sweating brow, or bloodied hands gripping stone.
    * **High Overhead Angle / Bird's Eye**: Used to frame solitary vulnerability against the wide environment.
- Art Style Presets by Scene Mood (Điều phối phong cách mỹ thuật theo cảm xúc):
  - **Atmospheric / Contemplative Preset (Quiet / Emotional / Transition Beats)**:
    For peaceful transitions and environmental atmosphere, describe: *Rich painterly background art, nuanced environmental lighting matching the literal time of day and weather, subtle atmospheric particle physics, soft filmic depth of field*.
  - **High-Impact Sakuga Preset (Action / Kinetic Combat)**:
    For battles, speed bursts, and awakening moments, describe: *High-budget Sakuga action style, sharp crisp cel shading, high contrast, luminous rim lighting, dynamic motion blur*.
- Anime Visual Effects & High-Speed Action Staging (Sakuga VFX):
  - **Canon Power & Attribute Integrity**:
    * Strictly obey the powers, elements, and physical limitations documented in the character roster. Do NOT invent foreign magical effects, auras, or energy types not belonging to the character.
  - Anime Frame Skip & Visual Ellipsis (Tỉnh lược khung hình & Nhịp độ Kinetic):
    * Action and hazard shots must be fast and snappy (1.2s–2.5s), NEVER lingering for excessive durations!
    * Direct authentic Japanese anime action shorthand (4-Phase Sakuga Grammar):
      1. **Pha 1 (Instinctive Push-off / Sudden Spring)**: Sudden adrenaline reflex, feet kicking off the ground, sudden shockwave burst of wind blowing dust and hair (`[AnimeFX: ActionBurst]`). Duration: ~1.5s.
      2. **Pha 2 (Frame Skip & Speed Streak)**: Tag `[AnimeFX: FrameSkip]` + `[AnimeFX: SpeedStreak]` (or `[AnimeFX: Vanish]`). The screen blurs into directional sonic air blur and clear wind streaks (`vệt tốc độ / vệt áp lực gió`), and the subjects vanish in a single frame. Duration: ~1.2s.
      3. **Pha 3 (Empty Impact / Near-Miss Momentum)**: Tag `[AnimeFX: SpeedLines]` + `[AnimeFX: ScreenShake]`. The hazard violently sweeps past the completely EMPTY zone with smoking skid marks or shockwave. Duration: ~2.0s.
      4. **Pha 4 (Kinetic Friction Skid & Momentum Absorption)**: Tag `[AnimeFX: ActionBurst]`. Low-angle shot along the ground: footwear slides hard creating friction smoke, body pivoting low to absorb momentum as shockwave ripples outward! Duration: ~1.8s.
      5. **Pha 5 (Touchdown Settle & Reaction)**: Settling into breath recovery, surrounded by settling dust. Duration: 3.5s–5.5s for dialogue/reaction.
    * Clash / Sonic Boom: Tag `[AnimeFX: Impact]` (sakuga monochrome flash) + `[AnimeFX: ScreenShake]`.
    * Critical Tension: Tag `[AnimeFX: SlowMo]` (bullet-time dilation) before explosive action snaps.
- Dialogue Attribution & Lip-Sync Integrity:
  - In shots with multiple characters present, clearly denote the SINGLE active speaking character: `[Speaker] <CharacterName> (<Emotion>): '<Line>'` or `[Thought] <CharacterName> (<Emotion>): '<Line>'`.
  - All non-speaking characters present in the frame must be explicitly described as listening or reacting silently with closed lips, preventing dialogue confusion.
- Spatial Continuity & Environmental Consistency:
  - All shots within the same scene must take place in the exact same physical location and lighting conditions. Camera angles may change, but the architectural background, walls, floor, props, and atmosphere must remain strictly coherent.
- CRITICAL JSON FORMATTING & VIETSUB RULE:
  - Wrap the valid JSON output inside a ```json fenced code block. Inside JSON string values, quote spoken lines with single quotes `'...'` (or properly escaped `\"`) and NEVER leave raw unescaped double quotes inside strings.
  - Whenever `<SCRIPT>` includes a `(Vietsub: ...)` line for a beat or dialogue, ALWAYS preserve `(Vietsub: ...)` verbatim inside that shot's `audio_desc` right after the dialogue/thought line.
"""


human_prompt_template_design_storyboard = \
"""
<SCRIPT>
{script_str}
</SCRIPT>

<CHARACTERS>
{characters_str}
</CHARACTERS>

<USER_REQUIREMENT>
{user_requirement_str}
</USER_REQUIREMENT>
"""



system_prompt_template_decompose_visual_description = \
"""
[Role]
You are a professional visual text analyst, proficient in cinematic language and shot narration. Your expertise lies in deconstructing a comprehensive shot description accurately into three core components: the static first frame, the static last frame, and the dynamic motion that connects them.

[Task]
Your task is to dissect and rewrite a user-provided visual text description of a shot strictly and insightfully into three distinct parts:
- First Frame Description: Describe the static image at the very beginning of the shot. Focus on compositional elements, initial character postures, environmental layout, lighting, color, and other static visual aspects.
- Last Frame Description: Describe the static image at the very end of the shot. Similarly, focus on the static composition, but it must reflect the final state after changes caused by camera movement or internal element motion.
- Motion Description: Describe all movements that occur between the first frame and the last frame. This includes camera movement (e.g., static, push-in, pull-out, pan, track, follow, tilt, etc.) and movement of elements within the shot (e.g., character movement, object displacement, changes in lighting, etc.). This is the most dynamic part of the entire description. For the movement and changes of a character, you cannot directly use the character's name to refer to them. Instead, you need to refer to the character by their external features, especially noticeable ones like clothing characteristics.

[Input]
You will receive a single visual text description of a shot that typically implicitly or explicitly contains information about the starting state, the motion process, and the ending state.
Additionally, you will receive a sequence of potential characters, each containing an identifier and a feature.
- The description is enclosed within <VISUAL_DESC> and </VISUAL_DESC>.
- The character list is enclosed within <CHARACTERS> and </CHARACTERS>.


[Output]
{format_instructions}

[Guidelines]
- Ensure all output values (except keys) match the language used in the script.
- Ensure the first and last frame descriptions are pure "snapshots," containing no ongoing actions (e.g., "He is about to stand up" is unacceptable; it should be "He is sitting on the chair, leaning slightly forward").
- In the motion description, you must clearly distinguish between camera movement and on-screen movement. Use professional cinematic terminology (e.g., dolly shot, pan, zoom, etc.) as precisely as possible to describe camera movement.
- In the motion description, you cannot directly use character names to refer to characters; instead, you should use the characters' visible characteristics to refer to them. For example, "Alice is walking" is unacceptable; it should be "Alice (short hair, wearing a green dress) is walking".
- The last frame description must be logically consistent with the first frame description and the motion description. All actions described in the motion section should be reflected in the static image of the last frame.
- If the input description is ambiguous about certain details, you may make reasonable inferences and additions based on the context to make all three sections complete and fluent. However, core elements must strictly adhere to the input text.
- Use accurate, concise, and professional descriptive language. Avoid overly literary rhetoric such as metaphors or emotional flourishes; focus on providing information that can be visualized.
- Similar to the input visual description, the first and last frame descriptions should include details such as shot type, angle, composition, etc.
- Below are the three types of variation within a shot (not between two shots):
(1) 'large' cases typically involve the exaggerated transition shots which means a significant change in the composition and focus, such as smoothly changing from a wide shot to a close-up. It is usually accompanied by significant camera movement (e.g., drone perspective shots across the city).
(2) 'medium' cases often involve the introduction of new characters and a character turns from the back to face the front (facing the camera).
(3) 'small' cases usually involve minor changes, such as expression changes, movement and pose changes of existing characters(e.g., walking, sitting down, standing up), moderate camera movements(e.g., pan, tilt, track).
- CRITICAL KEYFRAME ANCHORING RULE FOR ANIME ACTION: Any shot involving combat clashes, dodging, leaping, knocking back, falling, teleporting, projectile blasts, environmental destruction, or major pose/camera shifts MUST be labeled 'medium' or 'large' (NEVER 'small'), with clearly distinct starting pose (`ff_desc`) and climax/impact pose (`lf_desc`), so both `first_frame` and `last_frame` keyframes anchor the motion. Reserve 'small' strictly for quiet dialogue, subtle breathing, blinking, or gentle camera holds.
- CINEMATIC CAMERA ANGLE PRESERVATION: The first and last frame descriptions must strictly preserve dynamic angles (low-angle worm's-eye, canted Dutch angle, ground tracking). NEVER flatten or downgrade dynamic angles into generic 'medium shot' or 'eye-level shot' in ff_desc or lf_desc.
- When describing a character, it is necessary to indicate the direction they are facing.
- The first shot must establish the overall scene environment, using the widest possible shot.
- Use as few camera positions as possible.
- CRITICAL JSON FORMATTING: Wrap the valid JSON output inside a ```json fenced code block. Inside JSON string values, quote dialogue/narration with single quotes `'...'` (NEVER raw unescaped double quotes inside JSON strings).
"""


human_prompt_template_decompose_visual_description = \
"""
<VISUAL_DESC>
{visual_desc}
</VISUAL_DESC>

<CHARACTERS>
{characters_str}
</CHARACTERS>
"""


class VisDescDecompositionResponse(BaseModel):
    ff_desc: str = Field(
        description="A detailed description of the first frame of the shot, capturing the initial visual elements and composition.",
        # examples=[
        #     "Medium shot of a supermarket aisle at eye level. Bob(a tall man wearing a blue shirt and jeans) is positioned on the right side of the frame, captured in profile and facing right, while Alice(a young woman with short hair, wearing a green dress) is on the left, shown pushing a shopping cart with her gaze lowered toward the ground. They are arranged in a front-to-back spatial relationship. Shelves line both sides of the frame, and cool-toned fluorescent lighting from above washes over the scene. The vibrant colors of product packaging contrast with the metallic gray of the shopping cart, all contained within a stable, horizontally balanced composition.",
        #     "Extreme long shot. Aerial view from hundreds of meters above the ground. The boundless golden desert resembles undulating frozen waves, occupying the vast majority of the frame. At the very center of the image, a tiny, solitary explorer appears only as a faint dark speck, dragging a long, lonely trail of footprints behind him, stretching all the way to the edge of the frame.",
        #     "Medium shot at eye level angle. Designer A(with a beard, wearing a white suit) leans forward passionately, speaking emphatically. Product Manager B(with a beard, wearing a white T-shirt) sits with crossed arms, looking skeptical. Between them, Development Engineer C(brown hair, wearing a blue T-shirt) appears anxious, glancing between the two. Project Manager D(curly hair, wearing a red T-shirt) prepares to mediate, focusing on a whiteboard. Bright overhead lighting highlights their expressions, with a blurred whiteboard and glass wall in the background.",
        #     "A low-angle close-up shot captures the figure from below, framing him from the chest up. His face appears resolute and commanding, his eyes piercing as he speaks passionately. Flecks of saliva are visible, emphasizing his intensity. The overcast sky breaks with occasional light, casting him as a heroic, almost monumental figure against the gloom.",
        #     "An extremely close-up of an old, motionless pocket watch. Soft light highlights scratches on its brass case and the enamel dial with Roman numerals. The second hand remains fixed at 'VIII', casting a sharp shadow. A wrinkled finger gently touches the glass surface, evoking a tangible sense of stillness and time.",
        #     "An over-the-shoulder shot at eye level, positioned behind Character A(red hair, wearing a white T-shirt). The foreground, including A's shoulder and head, is softly blurred, directing focus onto Character B(with a beard, wearing a white T-shirt)'s face. B's subtle reactions—shifting from surprise to confusion, then to a glimmer of understanding—are clearly visible. The café background is gently blurred with warm lighting.",
        # ]
    )
    ff_vis_char_idxs: List[int] = Field(
        description="A list of indices of characters that are visible in the first frame of the shot, corresponding to the character list provided in the input.",
        examples=[[0], [1], [0, 1], []]
    )
    lf_desc: str = Field(
        description="A detailed description of the last frame of the shot, capturing the concluding visual elements and composition.",
    )
    lf_vis_char_idxs: List[int] = Field(
        description="A list of indices of characters that are visible in the last frame of the shot, corresponding to the character list provided in the input.",
        examples=[[0], [1], [0, 1], []]
    )
    motion_desc: str = Field(
        description="The motion description of the shot. Describe the dynamic visual changes within the shot (camera movement and the movement of elements within the frame)",
        examples=[
            "Static camera. Alice (short hair, wearing a green dress) is walking towards the camera.",
            "Dolly in from meidum shot to close-up. Bob (with a beard, wearing a white T-shirt) smiles to the camera.",
        ]
    )
    variation_type: Literal["large", "medium", "small"] = Field(
        description="Indicates the degree of change between the first frame and the last frame.",
    )
    variation_reason: str = Field(
        description="The reason for the variation type of the shot.",
        examples=[
            "This is a smooth transition shot from the sky to the ground. The content of the shot has changed significantly, so the variation type is large.",
            "Compared to the first frame, a new character appears in the last frame, and there are no significant changes in the composition. So the variation type is medium.",
            "Compared to the first frame, there are only minor changes in the composition. So the variation type is small.",
            "This shot only shows Alice speaking and the changes in her facial expressions, thus the variation type is small.",
        ],
    )



class StoryboardArtist:
    def __init__(
        self,
        chat_model: BaseChatModel,
    ):
        self.chat_model = chat_model


    @retry(stop=stop_after_attempt(3), after=after_func)
    async def design_storyboard(
        self,
        script: str,
        characters: List[CharacterInScene],
        user_requirement: Optional[str] = None,
        retry_timeout: int = 240,
    ) -> List[ShotBriefDescription]:

        class StoryboardResponse(BaseModel):
            storyboard: List[ShotBriefDescription] = Field(
                description="A complete storyboard of the scene, including the visual and audio description of each shot.",
            )

        script_str = script.strip()
        characters_str = "\n".join([f"Character {index}: {char}" for index, char in enumerate(characters)])
        user_requirement_str = user_requirement.strip() if user_requirement else ""

        parser = PydanticOutputParser(pydantic_object=StoryboardResponse)
        messages = [
            ('system', system_prompt_template_design_storyboard.format(format_instructions=parser.get_format_instructions())),
            ('human', human_prompt_template_design_storyboard.format(script_str=script_str, characters_str=characters_str, user_requirement_str=user_requirement_str)),
        ]
        chain = self.chat_model | parser
        response: StoryboardResponse = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )
        storyboard = response.storyboard

        return storyboard

    async def review_and_repair_storyboard(
        self,
        storyboard: List[ShotBriefDescription],
        characters: List[CharacterInScene],
        script: str,
        retry_timeout: int = 180,
    ) -> List[ShotBriefDescription]:
        """Run an LLM senior-director review pass to catch & repair semantic, wardrobe, alias, layout, and pacing defects."""
        import json as _json
        import logging
        from agents.anime_director_guard import AnimeDirectorGuard

        preliminary_hints = AnimeDirectorGuard.audit_storyboard_defects(storyboard, characters=characters)

        class StoryboardAuditResult(BaseModel):
            needs_repair: bool = Field(
                description="True if ANY defect was found (multi-pose clones, split panels, blank white void, wardrobe/prop drift vs CHARACTERS/SCRIPT, duplicate/phantom character aliases, monotonous 4+ identical cam_idx, or empty blur transition > 1.5s); False if clean."
            )
            issues: List[str] = Field(
                default_factory=list,
                description="Concise list of defects found, or empty list if needs_repair is False.",
            )
            repaired_storyboard: Optional[List[ShotBriefDescription]] = Field(
                default=None,
                description="Complete repaired storyboard (re-indexed 0..N-1) ONLY when needs_repair is True; set to null when needs_repair is False.",
            )

        parser = PydanticOutputParser(pydantic_object=StoryboardAuditResult)
        characters_str = "\n".join(
            [
                f"Character {c.idx} ({c.identifier_in_scene}): static={c.static_features}; dynamic={c.dynamic_features}"
                for c in characters
            ]
        )
        current_json = _json.dumps([s.model_dump() for s in storyboard], ensure_ascii=False, indent=2)
        hints_str = "\n".join(f"- {d}" for d in preliminary_hints) if preliminary_hints else "(none)"

        critic_system = (
            "You are a senior anime episode director auditing a draft storyboard BEFORE keyframe generation.\n"
            "Evaluate the ENTIRE storyboard dynamically against CHARACTERS, SCRIPT, and any PRELIMINARY_HINTS:\n"
            "1. NO multi-pose character clones or same-character distance+foreground duplicates in one frame — split into separate single-view sequential anime shots (re-index idx 0..N-1).\n"
            "2. NO split side-by-side comic inset panels or manga decorative radial ink/sparkle marks.\n"
            "3. NO stark blank white/light void backgrounds (unless transition_in == 'white_flash') — use softly blurred scene-consistent lighting/bokeh.\n"
            "4. STRICT character identity, wardrobe & prop consistency against CHARACTERS and SCRIPT: fix wrong outfits, stray outdoor props indoors, dropped/missing character names (e.g. broken \"inside 's house\" or \"of is visible\" -> restore the full `<Character>` name), or phantom/duplicate alias characters.\n"
            "5. Remove any clashing/hallucinated camera boilerplate that contradicts the scene setting or action (e.g. 'boots coiled in a low sprinter stance', 'pavement', 'vertical stone wall' in a calm house/room scene, or 'two-shot' on a solo shot); keep natural anime camera grammar.\n"
            "6. Cinematography & pacing: break up 4+ consecutive identical cam_idx shots with close-up/reverse/reaction angles; keep empty characterless speed/blur transition shots <= 1.5s.\n"
            "7. If repairing, preserve plot order, [RefBeat: ...] tags, [AnimeFX: ...] tags, `<Character>` angle-bracket tags verbatim, and verbatim dialogue + Vietsub lines.\n"
            "IMPORTANT: Return ONLY a single ```json ... ``` fenced code block matching the schema so `<Character>` tags are preserved intact.\n"
            "If EVERYTHING is clean and cinematic, set needs_repair=false, issues=[], and repaired_storyboard=null.\n"
            "{format_instructions}"
        )
        critic_human = (
            "=== PRELIMINARY_HINTS ===\n{hints_str}\n\n"
            "=== CURRENT_STORYBOARD ===\n{current_json}\n\n"
            "=== CHARACTERS ===\n{characters_str}\n\n"
            "=== SCRIPT ===\n{script_str}"
        )
        messages = [
            ("system", critic_system.format(format_instructions=parser.get_format_instructions())),
            (
                "human",
                critic_human.format(
                    hints_str=hints_str,
                    current_json=current_json,
                    characters_str=characters_str,
                    script_str=script.strip(),
                ),
            ),
        ]
        def _has_dropped_tag_grammar(shots_list: List[ShotBriefDescription]) -> bool:
            for sh in shots_list:
                v = sh.visual_desc or ""
                if re.search(r"(?:^|\s)'s\b|\bof\s+is\s+visible\b|\bfaces\s*\.", v) or re.search(
                    r"(?:^|\.\s+)(?:sits|stands|has|wears|looks|holds|turns)\b", v
                ):
                    return True
            return False

        import re

        try:
            chain = self.chat_model | parser
            response: StoryboardAuditResult = await asyncio.wait_for(
                chain.ainvoke(messages),
                timeout=retry_timeout,
            )
            if response.needs_repair and response.repaired_storyboard:
                repaired = response.repaired_storyboard
                if _has_dropped_tag_grammar(repaired):
                    # Retry once explicitly reminding the model to keep ```json fences so web DOM proxies don't eat <Name> tags
                    retry_msgs = messages + [
                        (
                            "human",
                            "Your previous JSON dropped a `<Character>` name (leaving broken grammar like \"'s \" or \"sits on\"). "
                            "Re-emit the repaired JSON inside a single ```json ... ``` code block and keep every character's full name intact.",
                        )
                    ]
                    retry_resp: StoryboardAuditResult = await asyncio.wait_for(
                        chain.ainvoke(retry_msgs),
                        timeout=retry_timeout,
                    )
                    if retry_resp.needs_repair and retry_resp.repaired_storyboard and not _has_dropped_tag_grammar(retry_resp.repaired_storyboard):
                        repaired = retry_resp.repaired_storyboard
                print(f"[storyboard_llm_repaired] Fixed {len(response.issues)} issue(s): {response.issues}")
                logging.info("Storyboard LLM critic repaired %d issue(s): %s", len(response.issues), response.issues)
                return repaired
            print("[storyboard_llm_clean] needs_repair=False")
        except Exception as exc:
            print(f"[storyboard_llm_fallback] {exc}")
            logging.warning("Storyboard self-review repair failed (%s); keeping deterministic guard output.", exc)
        return storyboard

    async def review_and_repair_shot_descriptions(
        self,
        shot_descriptions: List[ShotDescription],
        characters: List[CharacterInScene],
        retry_timeout: int = 180,
    ) -> List[ShotDescription]:
        """Run a scene-level LLM critic over decomposed shot_descriptions (ff_desc/lf_desc/motion_desc/char_idxs)."""
        import json as _json
        import logging

        if not shot_descriptions:
            return shot_descriptions

        class ShotDescriptionAuditResult(BaseModel):
            needs_repair: bool = Field(
                description="True if any ff_desc/lf_desc/motion_desc/variation_reason or ff_vis_char_idxs/lf_vis_char_idxs violates canonical <CHARACTERS> outfits/identities, duplicates alias characters, or retains comic artifacts."
            )
            issues: List[str] = Field(
                default_factory=list,
                description="Concise list of shot-level defects found, or empty list if clean.",
            )
            repaired_shots: Optional[List[ShotDescription]] = Field(
                default=None,
                description="Full corrected ShotDescription list ONLY when needs_repair is True; set to null when clean.",
            )

        parser = PydanticOutputParser(pydantic_object=ShotDescriptionAuditResult)
        characters_str = "\n".join(
            [
                f"idx={c.idx} | name={c.identifier_in_scene} | visible={c.is_visible} | static={c.static_features} | dynamic={c.dynamic_features}"
                for c in characters
            ]
        )
        shots_json = _json.dumps([s.model_dump() for s in shot_descriptions], ensure_ascii=False, indent=2)

        critic_system = (
            "You are a senior anime animation director auditing decomposed shot keyframes (ff_desc, lf_desc, motion_desc, ff_vis_char_idxs, lf_vis_char_idxs) before image generation.\n"
            "Check EVERY shot against the canonical CHARACTERS list:\n"
            "1. Wardrobe, hair, anatomy & prop consistency: ff_desc and lf_desc MUST match each character's canonical static_features and dynamic_features (no drifted outfit colors, no unscripted backpacks indoors, human vs animal form accurate).\n"
            "2. NO duplicate character aliases: if CHARACTERS has multiple alias entries for the same character, keep ONLY the primary named idx actually shown in ff_vis_char_idxs/lf_vis_char_idxs and remove any duplicate phantom second person from ff_desc/lf_desc/motion_desc.\n"
            "3. NO comic artifacts or dropped character names left behind in ff_desc/lf_desc/motion_desc/variation_reason: remove split panels, multiple overlapping poses, blank white/light void backgrounds, radial tension ink lines, sparkle marks, and restore any broken possessive `'s` or missing character name.\n"
            "4. When repairing, keep idx, is_last, cam_idx, visual_desc, audio_desc, transition_in, continuity_from_prev, target_duration_sec aligned with input shots (and keep character names intact).\n"
            "IMPORTANT: Return ONLY a single ```json ... ``` fenced code block matching the schema.\n"
            "If ALL shots are clean, set needs_repair=false, issues=[], and repaired_shots=null.\n"
            "{format_instructions}"
        )
        critic_human = "=== CHARACTERS ===\n{characters_str}\n\n=== SHOT_DESCRIPTIONS ===\n{shots_json}"
        messages = [
            ("system", critic_system.format(format_instructions=parser.get_format_instructions())),
            ("human", critic_human.format(characters_str=characters_str, shots_json=shots_json)),
        ]
        try:
            chain = self.chat_model | parser
            response: ShotDescriptionAuditResult = await asyncio.wait_for(
                chain.ainvoke(messages),
                timeout=retry_timeout,
            )
            if response.needs_repair and response.repaired_shots and len(response.repaired_shots) == len(shot_descriptions):
                print(f"[shot_description_llm_repaired] Fixed {len(response.issues)} issue(s): {response.issues}")
                logging.info("ShotDescription LLM critic repaired %d issue(s): %s", len(response.issues), response.issues)
                return response.repaired_shots
            print("[shot_description_llm_clean] needs_repair=False")
        except Exception as exc:
            print(f"[shot_description_llm_fallback] {exc}")
            logging.warning("ShotDescription self-review failed (%s); keeping original decomposed shots.", exc)
        return shot_descriptions




    @retry(stop=stop_after_attempt(3), after=after_func)
    async def decompose_visual_description(
        self,
        shot_brief_desc: ShotBriefDescription,
        characters: List[CharacterInScene],
        retry_timeout: int = 180,
    ) -> ShotDescription:
        parser = PydanticOutputParser(pydantic_object=VisDescDecompositionResponse)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt_template_decompose_visual_description),
                ('human', human_prompt_template_decompose_visual_description),
            ]
        )
        chain = prompt_template | self.chat_model | parser

        visual_desc = shot_brief_desc.visual_desc.strip()

        characters_str = "\n".join([f"{char.identifier_in_scene}: (static) {char.static_features}; (dynamic) {char.dynamic_features}" for char in characters])

        decomposition: VisDescDecompositionResponse = await asyncio.wait_for(
            chain.ainvoke(
                input={
                    "format_instructions": parser.get_format_instructions(),
                    "visual_desc": visual_desc,
                    "characters_str": characters_str,
                },
            ),
            timeout=retry_timeout,
        )

        validate_char_idxs(decomposition.ff_vis_char_idxs, len(characters), "ff_vis_char_idxs")
        validate_char_idxs(decomposition.lf_vis_char_idxs, len(characters), "lf_vis_char_idxs")

        return ShotDescription(
            idx=shot_brief_desc.idx,
            is_last=shot_brief_desc.is_last,
            cam_idx=shot_brief_desc.cam_idx,
            visual_desc=shot_brief_desc.visual_desc,
            variation_type=decomposition.variation_type,
            variation_reason=decomposition.variation_reason,
            ff_desc=decomposition.ff_desc,
            ff_vis_char_idxs=decomposition.ff_vis_char_idxs,
            lf_desc=decomposition.lf_desc,
            lf_vis_char_idxs=decomposition.lf_vis_char_idxs,
            motion_desc=decomposition.motion_desc,
            audio_desc=shot_brief_desc.audio_desc,
            transition_in=getattr(shot_brief_desc, "transition_in", "cut"),
            continuity_from_prev=getattr(shot_brief_desc, "continuity_from_prev", False),
            target_duration_sec=getattr(shot_brief_desc, "target_duration_sec", None),
        )


def validate_char_idxs(idxs, num_characters, field_name):
    """Reject LLM-emitted character indices outside [0, num_characters).

    Negative values would silently select the wrong character via Python
    indexing; out-of-range values would crash deep inside the render gather.
    Raising here lets the @retry on decompose_visual_description re-ask.
    """
    invalid = [idx for idx in idxs if idx < 0 or idx >= num_characters]
    if invalid:
        raise ValueError(
            f"{field_name} contains invalid character indices {invalid}; "
            f"valid range is 0..{num_characters - 1}"
        )
