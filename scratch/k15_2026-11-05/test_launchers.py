"""CPU-only queue regression checks; never touches remote state or GPUs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import hashlib

spec = importlib.util.spec_from_file_location('k15_queue', Path(__file__).with_name('k15_queue.py'))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.k13 = self.root/'k13.txt'
        self.k13.write_text(''.join(f'LANE DONE lane{i} 2026-11-05T00:00:00-07:00\n' for i in range(4)))
        self.write(q.initial(True))

    def write(self, tasks):
        (self.root/'queue_state.json').write_text(json.dumps(dict(
            pending=tasks, active={}, attempts={}, done=[], giveup=[], reuse_added=[])))

    def call(self, action='claim', lane=0, rc=None):
        return q.operate(self.root, self.k13, action, lane, rc)

    def test_count_order(self):
        self.assertEqual(len(q.initial()), 30)
        self.assertEqual(len(q.initial(False)), 20)
        self.assertEqual(q.initial()[0], 'bigkey sum2-t4 2718')
        self.assertEqual(q.initial()[5], 'bigkey sink2-t4 2718')

    def test_global_barrier(self):
        self.write(['bigkey sum2-t4 2718', 'krx rep2 2718'])
        self.assertEqual(self.call(), 'bigkey sum2-t4 2718')
        self.assertEqual(self.call(lane=1), 'WAIT')
        self.call('finish', rc=0)
        self.assertEqual(self.call(lane=1), 'krx rep2 2718')

    def test_late_reuse_and_append_once(self):
        self.write(['krx rep2 2718'])
        self.k13.write_text('LANE DONE lane0 2026-11-05T00:00:00-07:00\n')
        self.assertEqual(self.call(), 'WAIT')
        with self.k13.open('a') as f: f.write('GIVEUP par2 s3141 after attempt3 timestamp\n')
        self.assertEqual(self.call(), 'bigkey par2-t4 3141')
        self.call('finish', rc=0)
        self.assertEqual(self.call(), 'WAIT')
        state = json.loads((self.root/'queue_state.json').read_text())
        self.assertEqual(state['reuse_added'], ['bigkey par2-t4 3141'])

    def test_three_requeues_including_timeout(self):
        self.write(['bigkey sum2-t4 2718'])
        for _ in range(4):
            self.assertEqual(self.call(), 'bigkey sum2-t4 2718')
            self.call('finish', rc=124)
        self.assertEqual(self.call(), 'DONE')
        text = (self.root/'status.txt').read_text()
        self.assertEqual(text.count('REQUEUE '), 3)
        self.assertEqual(text.count('GIVEUP '), 1)

    def test_duplicate_lease_refused(self):
        self.call()
        with self.assertRaises(RuntimeError): self.call()


class GateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'code').mkdir()
        (self.root/'code/data').write_bytes(b'fixed K13 data')
        spec = importlib.util.spec_from_file_location('k15_gate', Path(__file__).with_name('gate.py'))
        self.gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.gate)
        self.m = dict(launch_approved=True,
            approval_reference='Jaden 2026-11-05: approve K15, launch it after K13 lands',
            sink='fixed', training_timeout_seconds=72*3600, scoring_timeout_seconds=7200,
            krx_supported=False, checks_passed=True, prereg_reference='registered',
            checks_reference='all checks passed', commit='a'*40,
            data_sha256={'data': hashlib.sha256(b'fixed K13 data').hexdigest()})
        self.path = self.root/'manifest.json'

    def validate(self, dirty=False):
        self.path.write_text(json.dumps(self.m))
        with mock.patch.object(self.gate.subprocess, 'check_output',
                               side_effect=['a'*40, 'dirty' if dirty else '']):
            return self.gate.validate(self.path)

    def test_approval_and_clean_exact_commit(self):
        self.assertEqual(self.validate()['commit'], 'a'*40)
        with self.assertRaises(AssertionError): self.validate(dirty=True)
        self.m['launch_approved'] = False
        with self.assertRaises(AssertionError): self.validate()

    def test_data_corruption_refused(self):
        (self.root/'code/data').write_bytes(b'changed')
        with self.assertRaises(AssertionError): self.validate()

    def test_unsupported_krx_and_sink_refused(self):
        self.m['krx_supported'] = True
        with self.assertRaises(AssertionError): self.validate()
        self.m['krx_supported'] = False
        self.m['sink'] = 'learned'
        with self.assertRaises(AssertionError): self.validate()


if __name__ == '__main__':
    unittest.main()
