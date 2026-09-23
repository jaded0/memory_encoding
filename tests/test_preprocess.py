"""Processed-dataset naming, missing data (auto-prepared locally, an error under SLURM), and old-vs-new batch equivalence (network-free)."""
import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset

import preprocess
from utils import collate_fn, filter_text, text_to_indices, text_to_indices_and_one_hot

HF_NAME = "jbrazzy/baby_names"  # text column "Names", full charset
TEXTS = ["Ann", "Bo", "Zoë-Lee\n", "", "O'Brien, Jr.", "x" * 40, "Émile", "Kai?", "Li", "Maximilian"]


def old_rows(texts, dataset_name):
    """Rows as the old pipeline stored them: text, int indices, one-hot per character."""
    dataset = Dataset.from_dict({"Names": texts})
    for fn in (filter_text, text_to_indices, text_to_indices_and_one_hot):
        dataset = dataset.map(fn, batched=True, fn_kwargs={"dataset_name": dataset_name})
    return list(dataset)


class ProcessedNameTest(unittest.TestCase):
    def test_name_carries_dataset_split_rows_charset_code_and_version(self):
        name = preprocess.processed_dataset_name("roneneldan/tinystories")
        self.assertTrue(name.startswith("roneneldan--tinystories__train__rows-1000000__charset-"), name)
        self.assertIn(f"__code-{preprocess.preprocessing_code_hash()}__", name)
        self.assertTrue(name.endswith(f"__v{preprocess.PREPROCESS_VERSION}"), name)
        self.assertNotIn("/", name)

    def test_anything_that_changes_the_rows_changes_the_name(self):
        base = preprocess.processed_dataset_name(HF_NAME)
        self.assertNotEqual(base, preprocess.processed_dataset_name(HF_NAME, limit=100))
        with patch.object(preprocess, "PREPROCESS_VERSION", preprocess.PREPROCESS_VERSION + 1):
            self.assertNotEqual(base, preprocess.processed_dataset_name(HF_NAME))
        with patch.object(preprocess, "get_charset", return_value="abc"):
            self.assertNotEqual(base, preprocess.processed_dataset_name(HF_NAME))
        with patch.object(preprocess, "preprocessing_code_hash", return_value="0000000000"):
            self.assertNotEqual(base, preprocess.processed_dataset_name(HF_NAME))
        self.assertEqual(base, preprocess.processed_dataset_name(HF_NAME))

    def test_data_dir_comes_from_env(self):
        with patch.dict(os.environ, {"EPHEMERAL_DATA_DIR": "/somewhere"}):
            self.assertEqual(preprocess.processed_data_dir(), "/somewhere")
        with patch.dict(os.environ, {"EPHEMERAL_DATA_DIR": ""}):
            self.assertEqual(preprocess.processed_data_dir(), os.path.join(preprocess.REPO_DIR, "processed_datasets"))

    def test_num_proc_follows_slurm_and_is_capped(self):
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "1", "SLURM_NTASKS": "10"}):
            self.assertEqual(preprocess.default_num_proc(), 1)
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "999"}), patch.object(os, "sched_getaffinity", return_value=set(range(64))):
            self.assertEqual(preprocess.default_num_proc(cap=16), 16)


@contextlib.contextmanager
def environment(data_dir, **variables):
    """os.environ with EPHEMERAL_DATA_DIR=data_dir and the given variables, and without
    SLURM_JOB_ID or EPHEMERAL_AUTO_PREPROCESS unless given."""
    with patch.dict(os.environ, {"EPHEMERAL_DATA_DIR": data_dir}):
        for name in ("SLURM_JOB_ID", preprocess.AUTO_PREPROCESS_ENV):
            os.environ.pop(name, None)
        os.environ.update(variables)
        yield


FUNCTION = '''
def filter_text(examples, dataset_name):
    """Filter out characters not in the charset."""
    key = dataset_keys.get(dataset_name)
    return {'text': [t for t in examples[key] if t]}
'''


class CodeHashTest(unittest.TestCase):
    def assert_same_code(self, edited, same):
        base = preprocess.normalised_function_source(FUNCTION)
        self.assertNotEqual(edited, FUNCTION)
        (self.assertEqual if same else self.assertNotEqual)(preprocess.normalised_function_source(edited), base)

    def test_comment_only_edits_keep_the_hash(self):
        self.assert_same_code(FUNCTION.replace("    key =", "    # which column holds the text\n    key =")
                              .replace("dataset_name)\n    return", "dataset_name)  # a trailing comment\n\n    return"),
                              same=True)

    def test_docstring_only_edits_keep_the_hash(self):
        self.assert_same_code(FUNCTION.replace("Filter out characters not in the charset.", "Something else entirely."), same=True)
        self.assert_same_code(FUNCTION.replace('    """Filter out characters not in the charset."""\n', ""), same=True)

    def test_code_edits_change_the_hash(self):
        for edited in (FUNCTION.replace("if t]", "if t.strip()]"),        # logic
                       FUNCTION.replace("'text'", "'texts'"),              # a string constant
                       FUNCTION.replace("key", "column"),                  # a name
                       FUNCTION.replace("dataset_keys.get(dataset_name)", "dataset_keys[dataset_name]")):
            with self.subTest(edited=edited):
                self.assert_same_code(edited, same=False)

    def test_the_real_hash_uses_the_normalised_utils_functions(self):
        import inspect
        expected = preprocess._short_hash("".join(
            preprocess.normalised_function_source(inspect.getsource(fn)) for fn in (filter_text, text_to_indices)))
        self.assertEqual(preprocess.preprocessing_code_hash(), expected)
        self.assertNotIn("Filter out characters", preprocess.normalised_function_source(inspect.getsource(filter_text)))


class MissingDataTest(unittest.TestCase):
    def assert_fails_with_setup_hint(self, **variables):
        with tempfile.TemporaryDirectory() as data_dir, environment(data_dir, **variables), \
                patch.object(preprocess, "load_dataset", side_effect=AssertionError("must not preprocess")):
            os.mkdir(os.path.join(data_dir, "jbrazzy--baby_names__train__rows-all__charset-x__code-y__v0"))
            with self.assertRaises(preprocess.ProcessedDatasetMissing) as caught, contextlib.redirect_stdout(io.StringIO()):
                preprocess.load_and_preprocess_data(HF_NAME, batch_size=2, seed=1)
            self.assertEqual(os.listdir(data_dir), ["jbrazzy--baby_names__train__rows-all__charset-x__code-y__v0"])
        message = str(caught.exception)
        self.assertIn(preprocess.processed_dataset_name(HF_NAME), message)
        self.assertIn("setup_cluster/prepare_datasets.sbatch", message)
        self.assertIn("python preprocess.py jbrazzy/baby_names", message)
        self.assertIn("stale", message)

    def test_a_slurm_job_fails_with_setup_hint_instead_of_preprocessing(self):
        self.assert_fails_with_setup_hint(SLURM_JOB_ID="12345")
        self.assert_fails_with_setup_hint(SLURM_JOB_ID="12345", EPHEMERAL_AUTO_PREPROCESS="")

    def test_auto_preprocess_0_fails_locally_too(self):
        self.assert_fails_with_setup_hint(EPHEMERAL_AUTO_PREPROCESS="0")

    def assert_prepares_then_loads(self, **variables):
        raw = Dataset.from_dict({"Names": TEXTS})
        with tempfile.TemporaryDirectory() as data_dir, environment(data_dir, **variables), \
                patch.object(preprocess, "load_dataset", return_value=raw) as load_raw:
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                loader = preprocess.load_and_preprocess_data(HF_NAME, batch_size=2, seed=1)
            # The same preparation as `python preprocess.py <name>`: the raw split, saved under
            # the processed name in the processed-data directory.
            load_raw.assert_called_once_with(HF_NAME, split=preprocess.dataset_keys[HF_NAME])
            path = preprocess.processed_dataset_path(HF_NAME)
            self.assertEqual(os.listdir(data_dir), [preprocess.processed_dataset_name(HF_NAME)])
            self.assertTrue(os.path.isfile(os.path.join(path, "preprocess_info.json")))
            self.assertIn("Preparing it now", printed.getvalue())
            loader.num_workers = 0
            self.assertEqual(sum(len(texts) for texts, _, _ in loader), len(TEXTS))
            # Present now, so the next run only loads it.
            load_raw.reset_mock()
            with contextlib.redirect_stdout(io.StringIO()):
                preprocess.load_and_preprocess_data(HF_NAME, batch_size=2, seed=1)
            load_raw.assert_not_called()

    def test_a_local_run_prepares_the_missing_dataset_and_continues(self):
        self.assert_prepares_then_loads()

    def test_auto_preprocess_1_prepares_even_under_slurm(self):
        self.assert_prepares_then_loads(SLURM_JOB_ID="12345", EPHEMERAL_AUTO_PREPROCESS="1")

    def test_auto_preprocess_rejects_other_values(self):
        with environment("/unused", EPHEMERAL_AUTO_PREPROCESS="maybe"):
            with self.assertRaises(ValueError):
                preprocess.auto_preprocess_enabled()

    def test_synthetic_datasets_are_not_prepared(self):
        with self.assertRaises(ValueError):
            preprocess.prepare_dataset("2_small_palindrome_dataset_vary_length")


class PrepareAndLoadTest(unittest.TestCase):
    def prepare(self, data_dir, num_proc):
        raw = Dataset.from_dict({"Names": TEXTS * 5})
        with patch.object(preprocess, "load_dataset", return_value=raw), contextlib.redirect_stdout(io.StringIO()):
            return preprocess.prepare_dataset(HF_NAME, num_proc=num_proc, data_dir=data_dir)

    def test_saved_rows_are_indices_and_identical_across_num_proc(self):
        with tempfile.TemporaryDirectory() as data_dir:
            serial = preprocess.load_from_disk(self.prepare(os.path.join(data_dir, "a"), num_proc=1))
            parallel = preprocess.load_from_disk(self.prepare(os.path.join(data_dir, "b"), num_proc=3))
            self.assertEqual(serial.column_names, ["text", "tensor"])
            self.assertEqual(serial.features["tensor"].feature.dtype, "uint8")
            self.assertTrue(serial.data.table.equals(parallel.data.table))
            self.assertTrue(os.path.isfile(os.path.join(data_dir, "a", preprocess.processed_dataset_name(HF_NAME), "preprocess_info.json")))

    def test_batches_match_the_old_one_hot_pipeline(self):
        with tempfile.TemporaryDirectory() as data_dir, patch.dict(os.environ, {"EPHEMERAL_DATA_DIR": data_dir}):
            self.prepare(data_dir, num_proc=1)
            with contextlib.redirect_stdout(io.StringIO()):
                loader = preprocess.load_and_preprocess_data(HF_NAME, batch_size=4, seed=5)
            loader.num_workers = 0
            old = old_rows(TEXTS * 5, HF_NAME)
            # Same shuffle and sampler as the new loader, over the old stored rows.
            order = Dataset.from_dict({"i": list(range(len(old)))}).shuffle(seed=5, keep_in_memory=True)["i"]
            old_loader = preprocess.make_dataloader([old[i] for i in order], 4, seed=5, num_workers=0)
            batches = 0
            for (old_text, old_idx, old_hot), (new_text, new_idx, new_hot) in zip(old_loader, loader):
                self.assertEqual(old_text, new_text)
                self.assertEqual((old_idx.dtype, old_hot.dtype), (new_idx.dtype, new_hot.dtype))
                self.assertTrue(torch.equal(old_idx, new_idx))
                self.assertTrue(torch.equal(old_hot, new_hot))
                batches += 1
            self.assertEqual(batches, len(old) // 4)

    def test_collate_pads_one_hot_with_zero_rows(self):
        rows = old_rows(["abc", "abcdefg"], HF_NAME)
        n_characters = len(rows[0]["onehot_tensor"][0])
        old = collate_fn(rows)
        new = preprocess.OneHotCollate(n_characters)([{"text": r["text"], "tensor": r["tensor"]} for r in rows])
        self.assertEqual(old[0], new[0])
        self.assertTrue(torch.equal(old[1], new[1]) and torch.equal(old[2], new[2]))
        self.assertEqual(new[2][0, 3:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
