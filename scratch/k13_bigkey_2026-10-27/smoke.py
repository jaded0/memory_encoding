"""Bounded local GPU0 plumbing smoke (run with timeout580); never a learning result."""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import torch
import bench_tasks

if __name__ == '__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'local smoke is GPU0 only'
    torch.cuda.set_per_process_memory_fraction(.25)  # 1080 Ti: <3 GiB allocator
    out = Path(sys.argv[1]).expanduser()
    bench_tasks.generate_dataset('mqar2_32_q4', sizes={'train': 256, 'validation': 128, 'test': 128})
    sys.argv = ['train.py', '--dataset', 'mqar2_32_q4', '--input_mode', 'last_two',
                '--forget_rate', '.01', '--ephemeral_update_clamp', '0', '--weight_clamp', '0',
                '--slow_weight_decay', '0', '--hidden_size', '1024', '--num_layers', '3',
                '--residual_connection', 'false', '--positional_encoding_dim', '0',
                '--batch_size', '16', '--track', 'false', '--deterministic', 'true',
                '--resume', 'false', '--model_type', 'ephemeral', '--updater', 'dfa',
                '--enable_recurrence', 'false', '--fused_update', 'false', '--learning_rate', '1e-4',
                '--plasticity', '3e4', '--ephemeral_fraction', '.1', '--optimizer', 'sgd',
                '--n_iters', '3', '--print_freq', '1', '--early_stop_window', '0',
                '--heldout_eval_every', '3', '--heldout_batches', '1',
                '--checkpoint_save_freq', '3', '--seed', '2718', '--checkpoint_dir', str(out)]
    runpy.run_path(str(ROOT/'train.py'), run_name='__main__')
    print('SMOKE_MAX_ALLOCATED_BYTES', torch.cuda.max_memory_allocated())
