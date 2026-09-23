"""Old flag names still parse (with a one-line note) and land under today's names."""
import contextlib
import io
import unittest

from train import parse_args
from utils import upgrade_legacy_config

RENAMED = {  # old flag -> (new dest, value given, parsed value)
    "--plast_clip": ("plasticity", "7", 7.0),
    "--plast_proportion": ("ephemeral_fraction", "0.3", 0.3),
    "--clip_weights": ("weight_clamp", "0.5", 0.5),
    "--normalize": ("unit_norm_weights", "True", True),
}
OLD_NAMES = {"plast_clip", "plast_proportion", "clip_weights", "normalize", "grad_clip",
             "plast_learning_rate", "imprint_rate"}


def parse(*argv):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        args = parse_args(list(argv))
    return vars(args), output.getvalue()


def parse_error(*argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with unittest.TestCase().assertRaises(SystemExit) as raised:
            parse_args(list(argv))
    return raised.exception.code


class DeprecatedFlagTest(unittest.TestCase):
    def test_old_names_set_the_new_settings_with_one_note_each(self):
        for old, (dest, value, parsed) in RENAMED.items():
            with self.subTest(flag=old):
                args, printed = parse(old, value)
                self.assertEqual(args[dest], parsed)
                new = "--" + dest
                self.assertEqual(printed.splitlines(), [f"DEPRECATED: {old} is now {new} (same meaning); the old name still works."])
                self.assertEqual(parse(new, value), (args, ""))

    def test_normalize_without_a_value_means_true(self):
        self.assertTrue(parse("--normalize")[0]["unit_norm_weights"])

    def test_grad_clip_sets_the_clip_the_model_type_uses(self):
        for model_type, target, other in (("ephemeral", "ephemeral_update_clamp", "grad_norm_clip"),
                                          ("rnn", "grad_norm_clip", "ephemeral_update_clamp")):
            with self.subTest(model_type=model_type):
                args, printed = parse("--model_type", model_type, "--grad_clip", "0.2")
                self.assertEqual((args[target], args[other]), (0.2, 0))
                self.assertEqual(len(printed.splitlines()), 1)
                self.assertIn(f"sets --{target}", printed)
                self.assertEqual(parse("--model_type", model_type, f"--{target}", "0.2")[0], args)

    def test_grad_clip_defaults_to_the_model_type_default(self):
        # --model_type ephemeral is the default, so a bare --grad_clip is the update clamp.
        self.assertEqual(parse("--grad_clip", "0.2")[0]["ephemeral_update_clamp"], 0.2)

    def test_removed_flags_are_accepted_and_ignored(self):
        args, printed = parse("--plast_learning_rate", "0.005", "--imprint_rate", "0")
        self.assertEqual(args, parse()[0])
        self.assertEqual(printed.splitlines(), [
            "DEPRECATED: --plast_learning_rate was unused and is now ignored; remove it.",
            "DEPRECATED: --imprint_rate was unused and is now ignored; remove it.",
        ])

    def test_no_old_name_reaches_the_config(self):
        argv = ["--plast_clip", "7", "--plast_proportion", "0.3", "--clip_weights", "0.5", "--normalize", "True",
                "--grad_clip", "0.2", "--plast_learning_rate", "0.005", "--imprint_rate", "0"]
        args = parse(*argv)[0]
        self.assertFalse(OLD_NAMES & set(args))
        self.assertFalse({key for key in args if key.startswith("_")})

    def test_conflicting_old_and_new_values_are_an_error(self):
        self.assertEqual(parse_error("--plasticity", "3", "--plast_clip", "4"), 2)
        self.assertEqual(parse_error("--ephemeral_update_clamp", "0.1", "--grad_clip", "0.2"), 2)
        self.assertEqual(parse_error("--model_type", "rnn", "--grad_norm_clip", "0.1", "--grad_clip", "0.2"), 2)
        # Agreeing values, or the other model's clip, are fine.
        self.assertEqual(parse("--plasticity", "3", "--plast_clip", "3")[0]["plasticity"], 3.0)
        args = parse("--grad_norm_clip", "0.1", "--grad_clip", "0.2")[0]
        self.assertEqual((args["ephemeral_update_clamp"], args["grad_norm_clip"]), (0.2, 0.1))


class LegacyConfigKeyTest(unittest.TestCase):
    def test_old_config_keys_map_to_what_the_old_flags_parse_to(self):
        old_argv = ["--plast_clip", "7", "--plast_proportion", "0.3", "--clip_weights", "0.5",
                    "--normalize", "True", "--grad_clip", "0.2", "--plast_learning_rate", "0.005",
                    "--imprint_rate", "0"]
        old_config = {"plast_clip": 7.0, "plast_proportion": 0.3, "clip_weights": 0.5, "normalize": True,
                      "grad_clip": 0.2, "plast_learning_rate": 0.005, "imprint_rate": 0.0}
        keys = ("plasticity", "ephemeral_fraction", "weight_clamp", "unit_norm_weights",
                "ephemeral_update_clamp", "grad_norm_clip")
        for model_type in ("ephemeral", "ethereal", "rnn"):
            with self.subTest(model_type=model_type):
                cli_model_type = "rnn" if model_type == "rnn" else "ephemeral"
                parsed = parse("--model_type", cli_model_type, *old_argv)[0]
                upgraded = upgrade_legacy_config({**old_config, "model_type": model_type})
                self.assertEqual({key: upgraded[key] for key in keys}, {key: parsed[key] for key in keys})
                self.assertFalse(OLD_NAMES & set(upgraded))

    def test_grad_clip_without_model_type_is_left_for_the_diff(self):
        self.assertEqual(upgrade_legacy_config({"grad_clip": 0.2}), {"grad_clip": 0.2})

    def test_new_config_is_unchanged(self):
        config = {"plasticity": 7.0, "ephemeral_update_clamp": 0.2, "grad_norm_clip": 0, "model_type": "ephemeral"}
        self.assertEqual(upgrade_legacy_config(config), config)


if __name__ == "__main__":
    unittest.main()
