"""test_memory_module.py — CPU unit tests for LearnedKeyMemory (run on every change; `python test_memory_module.py`).
Written after run 10 (2026-11-03), which was voided by a segment-carry off-by-one at win=1."""
import torch, torch.nn.functional as F, sys
from memory_module import LearnedKeyMemory
def test_windows_carry():
    torch.manual_seed(0); V, d = 50, 16; E = torch.randn(V, d)
    for win in (1, 2, 8):
        for ema in (0.0, 0.9):
            m = LearnedKeyMemory(d, E, win=win, dk=64, sparse=0.1, nh=1, ema=ema).eval(); B, L = 2, 12
            h = torch.randn(B, 2 * L, d); x = torch.randint(0, V, (B, 2 * L)); ex = m.E[x]
            st = m.init_state(B, "cpu"); w_whole = m._windows(h, ex, st)                       # one pass over 2L tokens
            st2 = m.init_state(B, "cpu"); w1 = m._windows(h[:, :L], ex[:, :L], st2); w2 = m._windows(h[:, L:], ex[:, L:], st2)   # two segments
            assert torch.allclose(torch.cat([w1, w2], 1), w_whole, atol=1e-5), f"segment carry mismatch win={win} ema={ema}"
            assert st2["hist"].shape[1] == win - 1, f"hist length {st2['hist'].shape[1]} != win-1 (win={win})"
    print("ok: windows carry across segments for win 1/2/8, ema off/on")
def test_forward_backward():
    torch.manual_seed(0); V, d = 50, 16; E = torch.randn(V, d); m = LearnedKeyMemory(d, E, win=1, dk=64, sparse=0.1, nh=2, ema=0.9); B, L = 2, 10
    h = torch.randn(B, L, 2 * d); x = torch.randint(0, V, (B, L)); lp = F.log_softmax(torch.randn(B, L, V), -1); st = m.init_state(B, "cpu")
    out, st = m(h, lp, x, st, targets=x[:, 1:]); (out.sum() + m.aux_nll.sum() + m.wlogit.sum()).backward(); assert out.shape == (B, L, V)
    print("ok: forward/backward with nh=2, ema, win=1")
if __name__ == "__main__": test_windows_carry(); test_forward_backward()
