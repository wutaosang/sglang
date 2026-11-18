Archived NPU DSV4 prefill graph validation evidence

Production fix: 4a169c1b0068d45a087005689b0c7644d3728bda

metadata-abi-failure.png: original QuantLightningIndexerV2 ABI error and a regression failure before the fix.
hccl-initialization-order.png: original metadata-after-HCCL failure and a positive control that initializes HCCL first.

These are browser screenshots of archived, unchanged log excerpts from 2026-11-17. They are not screenshots of a new failing run. Each .log file contains the displayed excerpt. manifest.json records original-source SHA-256 digests and excerpt line ranges.

The production fix preserves each metadata producer's ABI length and warms HCCL before attention metadata initializes.

This attachment branch carries evidence for the PR description. GSM8K has no completed result at the time these attachments were prepared.
