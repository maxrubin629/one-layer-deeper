# v3.7 — Last-state readout ablation

This candidate keeps the complete v3 factored orbit/workspace recurrence and
learned query-weighted trajectory computation, but sends only the final
trajectory key to the logits. The zero-weighted `queried` term preserves the
query branch in the compute graph while disabling learned depth selection.

Compared with unchanged v3, this isolates whether selecting among recurrent
depths is useful beyond reading the deepest available state. All modules,
state keys, tensor shapes, optimizer settings, and 4/64 train/evaluation loop
counts remain matched to v3.
