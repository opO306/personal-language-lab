# personal-language-lab

From Understanding.

A small research prototype for text relations through a lossless Korean jamo/byte port.
This is a curated stable P08 snapshot, not the complete live project.

## Included

- Byte-pinned shared schema, scalar codec and batching contracts.
- Fixed327 vocabulary: PAD,67 modern jamo, byte fallback and markers.
- Embedding32 -> GRU64, four4-way relation heads and one2-way allowed head.
- 30,450 parameters; target-free inference; explicit owner/episode/seed/mode.
- Optional valid-token mean head readout; final recurrent state preserved.
- Small handwritten fixtures: masking, empty input, isolation, BPTT and supervision.
- Passive observation and linear-margin reconstruction helpers.
- Existing Windows CPU/RAM test guardian and execution-file binding tools.

## Small-fixture reproduction

Use Python3.12 with an existing PyTorch2.7.1 and pytest9.1.1 environment.
No dependency/model installation was performed for publication.
From the repository root on Windows:

    python -B tools/bounded_check.py tests/test_vocabulary.py tests/test_encoder.py tests/test_supervision.py tests/test_training_inference_boundary_red.py tests/test_p08_valid_mean_readout.py tests/test_p08_signal_observer_small.py -q

The existing guardian limits six logical CPUs and an8GiB Job, and stops below4GiB
available RAM. It is Windows-only; other platforms need equivalent resource controls.
Model computation is CPU float32. Selected tests contain small synthetic strings/labels
and require no sealed dataset or checkpoint.

## Results and limitations

A saved one-seed development comparison used824 TRAIN /200 internal-development
examples, seed31101, Adam0.001, batch16 and five epochs (260updates).
Last-state and valid-mean models both predicted [0,3,0,3,0] on every development example.
Joint all-five accuracy was **0%**. Allowed accuracy was79.5%, equal to always-false.
Valid mean did not improve accuracy in that bounded run.

A later18-forward fixed-eight observation found measurable mean-readout/logit differences
although argmax stayed equal. See [observations](docs/observations/development_summary.md)
and [signal chart](docs/observations/fixed8_signal.png).
These development observations were chosen after earlier results were inspected.
They are not independent final generalization or production-readiness results.
There is no dialogue decoder, open-ended text generation, shared memory or breeding.

## Publication boundary

Live M-A four-seed/team work is **in progress and excluded**, not reported complete.
Native77/current-base, historical B and binary original engines/archives are excluded.
Formal datasets, sealed targets, checkpoints, raw failure logs, caches, private plans,
photos/screenshots, identity/contact data and authentication files are not shipped.
Original artifacts remain local.

The actual pilot/chart cannot be regenerated from this snapshot: the private campaign
loader, full packs and checkpoints are deliberately excluded. Public fixtures verify
component contracts, not the full pilot. The actual campaign entry is removed from
the published observer module.

No new model calculation or training was run during publication because M-A owned the
computation slot. Publication validation is static syntax/provenance/privacy screening.
Earlier guarded checks apply to recorded source versions, not a newly tested public checkout.

See [PROVENANCE.json](PROVENANCE.json) for lineage/intentional publication edits.
No new project license is assigned; see [license status](LICENSE_STATUS.md).
