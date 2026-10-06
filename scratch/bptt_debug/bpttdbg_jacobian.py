"""Per-step recurrent Jacobian dh_t/dh_{t-1} of SimpleRNN at init, and hidden-state magnitude."""
import sys, torch
sys.path.insert(0, '.')
from ephemeral_model import SimpleRNN
torch.manual_seed(0)
n = 9
for hidden in (128, 1024):
    for layers in (1, 2, 3):
        for residual in (False, True):
            m = SimpleRNN(n, hidden, n, layers, dropout_rate=0, enable_recurrence=True, updater='bptt',
                          residual_connection=residual)
            x = torch.zeros(1, n); x[0, 3] = 1
            h = m.initHidden(1)
            for _ in range(3):  # a few steps from zero
                _, h = m(x, h)
            h0 = h.detach().clone().requires_grad_(True)
            J = torch.autograd.functional.jacobian(lambda hh: m(x, hh)[1], h0)[0, :, 0, :]
            s = torch.linalg.svdvals(J)
            # How much the input char changes the next hidden state (the write signal).
            x2 = torch.zeros(1, n); x2[0, 5] = 1
            dh = (m(x, h0)[1] - m(x2, h0)[1]).norm()
            print(f"hidden {hidden:5d} layers {layers} residual {residual!s:5}: |h| {h.norm():.3e} "
                  f"sigma_max(J) {s[0]:.3e}  |J|_F {J.norm():.3e}  write |dh| {dh:.3e}")
