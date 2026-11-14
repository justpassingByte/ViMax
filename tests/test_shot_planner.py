import unittest
from types import SimpleNamespace

from agents.shot_planner import (
    infer_beat_type,
    plan_scene,
    check_variety,
    load_rules,
    fx_tokens_to_tags,
    strip_forbidden_fx_tags,
)

RULES = load_rules()


def _beat(bid, action="", sfx=None, dialogues=None, is_break=False, present=None, beat_type=None):
    return SimpleNamespace(
        beat_id=bid, action_desc=action, sfx=sfx or [], dialogues=dialogues or [],
        is_scene_break=is_break, characters_present=present or [], beat_type=beat_type,
    )


def _dlg(speaker, line_type="spoken"):
    return SimpleNamespace(speaker=speaker, line_type=line_type)


class InferBeatTypeTests(unittest.TestCase):
    def test_explicit_wins(self):
        self.assertEqual(infer_beat_type(_beat("b", beat_type="reveal"), None, RULES), "reveal")

    def test_establishing_and_impact_and_action(self):
        self.assertEqual(infer_beat_type(_beat("b0", is_break=True), None, RULES), "establishing")
        self.assertEqual(infer_beat_type(_beat("b1", sfx=["CRASH!"]), _beat("p"), RULES), "impact")
        self.assertEqual(infer_beat_type(_beat("b2", action="he dashes forward"), _beat("p"), RULES), "action")

    def test_dialogue_exchange_monologue(self):
        prev = _beat("p", dialogues=[_dlg("Alex")])
        self.assertEqual(infer_beat_type(_beat("b", dialogues=[_dlg("Bea")]), prev, RULES), "exchange")
        self.assertEqual(infer_beat_type(_beat("b", dialogues=[_dlg("Alex")]), prev, RULES), "dialogue")
        self.assertEqual(infer_beat_type(_beat("b", dialogues=[_dlg("Alex", "thought")]), None, RULES), "monologue")

    def test_reaction_and_transition(self):
        self.assertEqual(infer_beat_type(_beat("b", present=["Alex"]), _beat("p"), RULES), "reaction")
        self.assertEqual(infer_beat_type(_beat("b"), _beat("p"), RULES), "transition")


class PlanSceneTests(unittest.TestCase):
    def _scene(self, n):
        beats = []
        for i in range(n):
            if i == 0:
                beats.append(_beat(f"b{i}", is_break=True))
            elif i % 3 == 0:
                beats.append(_beat(f"b{i}", action="he leaps and strikes", sfx=["BOOM!"]))
            else:
                beats.append(_beat(f"b{i}", dialogues=[_dlg("Alex" if i % 2 else "Bea")], present=["Alex"]))
        return beats

    def test_reproducible_with_seed(self):
        beats = self._scene(8)
        a = plan_scene(beats, RULES, seed=42)
        b = plan_scene(beats, RULES, seed=42)
        self.assertEqual([s.model_dump() for s in a], [s.model_dump() for s in b])

    def test_no_variety_violations_for_3_8_20(self):
        for n in (3, 8, 20):
            slots = plan_scene(self._scene(n), RULES, seed=7)
            self.assertEqual(len(slots), n)
            viol = check_variety(slots, RULES)
            # repair should remove same size+angle runs
            run_viol = [v for v in viol if "repeated more than" in v]
            self.assertEqual(run_viol, [], msg=f"n={n}: {run_viol}")


class FxTests(unittest.TestCase):
    def test_fx_mapping(self):
        tags, trans = fx_tokens_to_tags(["impact", "white_flash"], RULES)
        self.assertIn("[AnimeFX: Impact]", tags)
        self.assertEqual(trans, "white_flash")

    def test_strip_forbidden_fx(self):
        # dialogue beats disallow fx -> tag removed
        out = strip_forbidden_fx_tags("A talks [AnimeFX: Impact] calmly", "dialogue", RULES)
        self.assertNotIn("AnimeFX", out)
        # action allows fx -> kept
        keep = strip_forbidden_fx_tags("A strikes [AnimeFX: SpeedLines]", "action", RULES)
        self.assertIn("AnimeFX", keep)


if __name__ == "__main__":
    unittest.main()
