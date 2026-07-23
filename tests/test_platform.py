import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "marinegym" / "_platform.py"
SPEC = importlib.util.spec_from_file_location("marinegym_platform", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
resolve_experience_path = MODULE.resolve_experience_path


class ResolveExperiencePathTest(unittest.TestCase):
    def test_prefers_exp_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            apps = Path(temp_dir) / "custom-apps"
            apps.mkdir()
            experience = apps / "omni.isaac.sim.python.kit"
            experience.touch()

            result = resolve_experience_path(
                {"EXP_PATH": str(apps), "ISAACSIM_PATH": str(Path(temp_dir) / "isaac")}
            )

            self.assertEqual(Path(result), experience)

    def test_falls_back_to_isaac_sim_apps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            isaac_sim = Path(temp_dir) / "isaac-sim"
            apps = isaac_sim / "apps"
            apps.mkdir(parents=True)
            experience = apps / "omni.isaac.sim.python.kit"
            experience.touch()

            result = resolve_experience_path({"ISAACSIM_PATH": str(isaac_sim)})

            self.assertEqual(Path(result), experience)

    def test_reports_missing_configuration(self):
        with self.assertRaisesRegex(RuntimeError, "Set EXP_PATH"):
            resolve_experience_path({})


if __name__ == "__main__":
    unittest.main()
