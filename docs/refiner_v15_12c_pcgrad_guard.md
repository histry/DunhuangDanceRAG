# Refiner V15.12c: PCGrad and fixed dual-track guard

V15.12b established that all 92 fitted TRAIN contexts failed at the first
independent fit-context audit. The decoded edit amplitude remained near zero,
and endpoint/temporal gradients were repeatedly opposed. Therefore V15.13-P
phase features are still deferred: the network has not yet demonstrated that
it can fit contexts already used for updates.

V15.12c changes only the nonpublishing bridge learnability diagnostic. It keeps
the model architecture, RMS group objective, 400-step budget, decoder caps,
physical/fixed-support/fidelity/boundary audits, and exact endpoint/temporal
0.03 acceptance requirements unchanged.

The update direction uses symmetric PCGrad for the endpoint and temporal
scientific objectives. When their gradients conflict, each is projected off
the other before averaging. The common gradient is scaled to the RMS of the
two original task-gradient norms so cancellation cannot silently collapse the
output. The remaining physical, trust, and clean-fidelity gradient is included
only up to the largest scale that preserves a nonnegative directional product
with both scientific gradients. Every update still goes through the existing
deterministic loss closure and bounded backtracking.

The guard no longer intersects independently attained historical minima.
Endpoint, temporal, support, penetration, jerk, root-vertical, and clean
fidelity components use one immutable initial TRAIN anchor. Joint feasibility
and complete group loss use best-so-far references. Physical allowances are
absolute normalized TRAIN-objective epsilons, with tighter single-recording and
wider cross-event envelopes. They never accumulate because all component
comparisons remain anchored to the initial state. These diagnostic allowances
are not meters or production thresholds and do not modify any final gate.

For the first 50 checked transactions, only the zero-initialized output head
uses a 10x learning rate. The backbone remains at the configured rate, and all
trial steps are still subject to line search and guard rejection. Logs record
raw, masked, tapered, and applied tangent RMS/max values along with PCGrad
cosine, norms, amplitude scale, remainder scale, and final task directional
products.

Run `scripts/run_refiner_v15_12c_server.sh` only on the server at the exact
commit. It performs server validation before starting a fresh diagnostic. It
does not start a pilot, resume formal training, promote a checkpoint, or run
V15.13-P.
