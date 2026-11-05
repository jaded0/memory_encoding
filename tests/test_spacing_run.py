"""Bounded CPU integration tests for the standalone SP1 runner (no production run)."""
import copy
from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import spacing_run as run
from ephemeral_model import EphemeralRNN
from kv_tasks import CHARSET
from reproducibility import seed_everything


class ToyModel(torch.nn.Module):
    """Instrumented fast writer; the event log survives private-model deepcopy."""
    events = []

    def __init__(self, batch_size=2):
        super().__init__()
        self.slow = torch.nn.Parameter(torch.tensor(7.0))
        self.register_buffer("fast", torch.zeros(batch_size))
        self.batch_size = batch_size
        self.multiplier = 1

    def start_sequence_wipe(self):
        self.events.append(("consolidate",))
        self.wipe_fast_state()

    def wipe_fast_state(self):
        self.events.append(("wipe_all",))
        self.fast.zero_()

    def wipe_bands(self, bands):
        self.events.append(("wipe_band", tuple(bands)))
        self.fast.zero_()

    @contextmanager
    def band_forget_override(self, rates):
        for item in rates.items():
            self.events.append(("forget", *item))
        yield self

    def initHidden(self, batch):
        self.events.append(("row",))
        return torch.zeros(batch, 1)

    def forward(self, inputs, hidden):
        self.events.append(("forward", tuple(inputs[:, :len(CHARSET)].argmax(1).tolist())))
        output = torch.zeros(self.batch_size, len(CHARSET))
        output[:, 0] = self.fast
        return output, hidden

    def fast_only_dfa_step(self, *args):
        self.events.append(("write",))
        self.fast.add_(1)


class SpacingRunnerTests(unittest.TestCase):
    def args(self, *extra):
        return run.build_parser().parse_args([
            "--device", "cpu", "--hidden_size", "8", "--layers", "1", "--batch_size", "2",
            "--train_intervals", "1,2", "--train_delays", "0", "--eval_grid", "1,2",
            "--eval_delay", "0", "--eval_streams", "3", "--forget_multipliers", "1",
            "--repetitions", "2", "--log_every", "0", *extra])

    def test_paired_cells_exact_stream_count_and_no_training_mutation(self):
        args = self.args()
        config = run.make_config(args)
        model = ToyModel()
        saved = copy.deepcopy(model.state_dict())
        ToyModel.events = []
        with patch("sys.stdout", new=io.StringIO()):
            result = run.evaluate(model, config, args)
        self.assertEqual(len(result["cells"]), 6)
        cells = result["cells"]
        signature = lambda c: [(o["stream"], o["context"], o["query"], o["answer"])
                               for o in c["outcomes"]]
        for cell in cells:
            self.assertEqual(cell["streams"], 3)
            self.assertEqual(sum(cell["counts"].values()), 3)
            self.assertEqual(signature(cell), signature(cells[0]))
        self.assertEqual(sum(e[0] == "consolidate" for e in ToyModel.events), 1)
        # Two I values, two groups each: prefix L-1=3 once, then three query forks.
        self.assertEqual(sum(e[0] == "row" for e in ToyModel.events), 2 * 2 * (3 + 3))
        self.assertEqual(result["cost"]["sequence_rows"], 48)
        self.assertTrue(all(torch.equal(v, saved[k]) for k, v in model.state_dict().items()))
        self.assertTrue(model.training)
        self.assertEqual(model.multiplier, 1)

    def test_wipe_is_before_query_row_not_before_answer(self):
        args = self.args("--eval_grid", "1", "--eval_streams", "2",
                         "--eval_protocols", "wipe_band0")
        config = run.make_config(args)
        ToyModel.events = []
        with patch("sys.stdout", new=io.StringIO()):
            run.evaluate(ToyModel(), config, args)
        events = ToyModel.events
        index = events.index(("wipe_band", (0,)))
        self.assertEqual(events[index + 1], ("row",))
        self.assertEqual(events[index + 2][0], "forward")
        # Eight prequential writes per row, query answer is the last prediction.
        self.assertEqual(sum(e[0] == "write" for e in events[index:]), 8)
        self.assertEqual(sum(e[0] == "row" for e in events[:index]), 2)
        self.assertEqual(events[-1], ("write",))

    def test_multiplier_only_band0_and_oneband_protocols(self):
        args = self.args("--arm", "oneband", "--eval_grid", "1", "--eval_streams", "1",
                         "--forget_multipliers", "0.3333333333333333,1,3")
        config = run.make_config(args)
        ToyModel.events = []
        with patch("sys.stdout", new=io.StringIO()):
            result = run.evaluate(ToyModel(), config, args)
        self.assertEqual({c["protocol"] for c in result["cells"]}, {"no_wipe", "wipe_all"})
        self.assertEqual({e[1] for e in ToyModel.events if e[0] == "forget"}, {0})
        self.assertEqual(config["forget_rate"], 0.03)
        args.eval_protocols = ["wipe_band0"]
        with self.assertRaisesRegex(ValueError, "one-band"):
            run.evaluate(ToyModel(), config, args)

    def test_arm_configs_and_horizon_validation(self):
        for arm in ("shared", "separate", "flat", "oneband"):
            config = run.make_config(self.args("--arm", arm))
            self.assertEqual(config["write_gain_power"], -1 if arm == "flat" else 0)
            self.assertEqual(config["band_feedback"], "separate" if arm == "separate" else "shared")
            self.assertEqual(config["wipe_every"], config["spacing_horizon"])
            if arm == "oneband":
                self.assertTrue(config["plasticity_bands"].startswith("0.1:"))
        with self.assertRaisesRegex(ValueError, "horizon"):
            run.make_config(self.args("--horizon", "2"))

    def test_custom_wipe_and_inference_override_flags(self):
        args = self.args("--eval_grid", "1", "--eval_streams", "1", "--eval_only",
                         "--eval_protocols", "no_wipe", "--eval_wipe_bands", "0,1",
                         "--forget_rate_override", "0:0.02,1:0.005")
        config = run.make_config(args)
        original = copy.deepcopy(config)
        ToyModel.events = []
        with patch("sys.stdout", new=io.StringIO()):
            result = run.evaluate(ToyModel(), config, args)
        self.assertEqual([cell["protocol"] for cell in result["cells"]], ["no_wipe", "wipe_bands_0_1"])
        self.assertIn(("wipe_band", (0, 1)), ToyModel.events)
        self.assertIn(("forget", 0, 0.02), ToyModel.events)
        self.assertIn(("forget", 1, 0.005), ToyModel.events)
        self.assertEqual(config["plasticity_bands"], original["plasticity_bands"])
        self.assertEqual(config["forget_rate"], original["forget_rate"])
        self.assertEqual(result["forget_rate_override"], {"0": 0.02, "1": 0.005})
        args.forget_multipliers = "1,3"
        with self.assertRaisesRegex(ValueError, "multipliers 1"):
            run.evaluate(ToyModel(), config, args)
        args.forget_multipliers = "1"
        args.eval_wipe_bands = (2,)
        with self.assertRaisesRegex(ValueError, "outside configured"):
            run.evaluate(ToyModel(), config, args)
        with self.assertRaisesRegex(ValueError, "requires --eval_only"):
            run.main(["--forget_rate_override", "0:0.02", "--forget_multipliers", "1"])

    def test_invalid_intervention_flags_and_cuda_only_fused_update(self):
        for flag, bad_values in (("--eval_wipe_bands", ("-1", "0,0", "a")),
                                 ("--forget_rate_override", ("0:nan", "0:1.1", "-1:0.1", "0:0.1,0:0.2"))):
            for value in bad_values:
                with self.subTest(flag=flag, value=value), patch("sys.stderr", new=io.StringIO()), \
                        self.assertRaises(SystemExit):
                    self.args(flag, value)
        args = self.args("--fused_update")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            run.configure_runtime(ToyModel(), args)
        fake = Mock()
        fake.parameters.return_value = iter([SimpleNamespace(device=SimpleNamespace(type="cuda"))])
        run.configure_runtime(fake, args)
        fake.enable_fused_update.assert_called_once_with(compile=True)
        self.assertFalse(self.args().fused_update)

    def test_checkpoint_resume_identical_online_training(self):
        from train import build_model, train_batch
        args = self.args("--arm", "flat")
        config = run.make_config(args)
        seed_everything(9)
        with patch("sys.stdout", new=io.StringIO()):
            model = build_model(config, CHARSET, len(CHARSET))
        stream = run.make_stream(config)
        state = {"sequence_count": 0, "training_instance": 0}
        # Save mid-stream, not just at a wipe boundary.
        for _ in range(2):
            ids, hot = run.collate(next(stream), "cpu")
            train_batch(ids, hot, model, config, state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            with patch("sys.stdout", new=io.StringIO()):
                run.save_checkpoint(path, model, config, state, stream)
                resumed, rc, rs, online = run.load_checkpoint(path)
            self.assertEqual(rs, state)
            for _ in range(3):
                rows = next(stream)
                self.assertEqual(rows, next(online))
                ids, hot = run.collate(rows, "cpu")
                train_batch(ids, hot, model, config, state)
                train_batch(ids, hot, resumed, rc, rs)
            self.assertEqual(state, rs)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0,
                                           equal_nan=True, msg=key)

    def test_cli_train_save_eval_only_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "pilot.pt")
            output = str(Path(directory) / "metrics.json")
            options = ["--device", "cpu", "--hidden_size", "8", "--layers", "1",
                       "--batch_size", "1", "--train_intervals", "1", "--train_delays", "0",
                       "--eval_grid", "1", "--eval_delay", "0", "--eval_streams", "1",
                       "--repetitions", "2", "--forget_multipliers", "1",
                       "--eval_protocols", "no_wipe", "--json", output, "--log_every", "0"]
            with patch("sys.stdout", new=io.StringIO()):
                run.main(options + ["--train_steps", "2", "--skip_eval", "--checkpoint", checkpoint])
                self.assertFalse(Path(output).exists())
                result = run.main(options + ["--load", checkpoint, "--eval_only"])
                self.assertEqual(result["training_rows"], 2)
                self.assertEqual(json.loads(Path(output).read_text()), result)
                run.main(options + ["--load", checkpoint, "--train_steps", "4", "--eval_every", "1",
                                    "--save_every", "1", "--checkpoint", checkpoint])
                _, _, state, stream = run.load_checkpoint(checkpoint)
            self.assertEqual(state["sequence_count"], 4)
            self.assertEqual(stream.sequence_count, 4)

    def test_all_arms_query_forks_are_order_independent(self):
        from train import build_model, train_batch
        for arm in ("shared", "separate", "flat", "oneband"):
            with self.subTest(arm=arm), patch("sys.stdout", new=io.StringIO()):
                args = self.args("--arm", arm, "--eval_grid", "1", "--eval_streams", "1")
                config = run.make_config(args)
                model = build_model(config, CHARSET, len(CHARSET))
                ids, hot = run.collate(next(run.make_stream(config)), "cpu")
                train_batch(ids, hot, model, config, {"sequence_count": 0, "training_instance": 0})
                saved = run._model_state(model)
                rng = torch.get_rng_state().clone()
                first = run.evaluate(model, config, args)
                args.eval_protocols = [cell["protocol"] for cell in reversed(first["cells"])]
                second = run.evaluate(model, config, args)
                by_protocol = lambda result: {cell["protocol"]: cell["outcomes"] for cell in result["cells"]}
                self.assertEqual(by_protocol(first), by_protocol(second))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key, value in run._model_state(model).items():
                    torch.testing.assert_close(value, saved[key], rtol=0, atol=0, equal_nan=True)

    @unittest.skipUnless(hasattr(EphemeralRNN, "wipe_bands"), "band model worker not integrated yet")
    def test_real_model_frozen_slow_and_result_json(self):
        from train import build_model
        args = self.args("--eval_grid", "1", "--eval_streams", "1")
        config = run.make_config(args)
        with patch("sys.stdout", new=io.StringIO()):
            model = build_model(config, CHARSET, len(CHARSET))
        saved = copy.deepcopy(model.state_dict())
        original_step = type(model).fast_only_dfa_step
        slow_snapshots = []

        def checked_step(working, *a, **kw):
            layers = [*working.linear_layers, working.i2h, working.i2o]
            before = [(l.per_sample_weights.detach().clone(), l.bias.detach().clone()) for l in layers]
            original_step(working, *a, **kw)
            for layer, (weights, bias) in zip(layers, before):
                mask = ~layer.ephemeral_mask
                self.assertTrue(torch.equal(weights[:, mask], layer.per_sample_weights[:, mask]))
                self.assertTrue(torch.equal(bias, layer.bias))
            slow_snapshots.append(True)

        with patch.object(type(model), "fast_only_dfa_step", checked_step), \
                patch("sys.stdout", new=io.StringIO()):
            result = run.evaluate(model, config, args)
        self.assertTrue(slow_snapshots)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, saved[key]), key)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            run._atomic_json(path, result)
            self.assertEqual(json.loads(path.read_text()), result)


if __name__ == "__main__":
    unittest.main()
