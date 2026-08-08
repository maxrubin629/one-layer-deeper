# v3.2 — Local state refiner

This controlled child of v3.1 adds one small real-valued residual refiner after
the damped correction update.  It acts only on `e_t`; it cannot contract or
overwrite the persistent orbit.  The refiner is deliberately shallow so this
experiment measures local error repair rather than additional recurrent depth.
