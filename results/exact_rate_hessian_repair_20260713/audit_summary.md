# Existing-result exact-payload audit

This artifact audits already committed endpoints. It does not turn the old nominal-rate experiments into new exact-rate evidence.

| run | method | nominal | entropy lower bound | CSR16 | PPL delta |
|---|---:|---:|---:|---:|---:|
| Qwen2-7B attention | Q | 0.250000 | 0.250279 | 0.250279 | -0.099638 |
| Qwen2-7B attention | Q+L | 0.257812 | 0.258092 | 0.258092 | -0.536973 |
| Qwen2-7B attention | Q+S | 0.258000 | 0.262480 | 0.266837 | +0.443284 |
| Qwen2-7B attention | Q+S+L | 0.257674 | 0.261260 | 0.264511 | +0.165230 |
| Qwen2-7B attn+MLP | Q | 0.250000 | 0.250176 | 0.250176 | +0.384142 |
| Qwen2-7B attn+MLP | Q+L | 0.257950 | 0.258126 | 0.258126 | +0.028776 |
| Qwen2-7B attn+MLP | Q+S | 0.258000 | 0.262377 | 0.266527 | -0.276042 |
| Qwen2-7B attn+MLP | Q+S+L | 0.257963 | 0.261446 | 0.264491 | -0.280689 |

Q+S+L ranks first within Q/Q+L/Q+S/Q+S+L in 2/4 committed runs, but the winning margin is not stable.

The Q/S interaction is predominantly negative. That is useful error cancellation, but it must not be reported as Hessian orthogonality.

Sparse methods exceed the nominal rate after storing FP16 residual values and support. Exact-rate reruns must reduce nnz and re-evaluate NLL/PPL.
Qwen attention-only currently favors Q+L (PPL delta -0.536973); its payload correction is small because no sparse support is stored.
Qwen attn+MLP Q+S+L beats Q+S by only 0.004646 PPL before exact-rate rerunning.
