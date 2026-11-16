# Vision artifact and metric contracts

These contracts protect the released KSDD2/AD2 evidence. They do not authorise retraining, protected-data access, model promotion or cloud work.

## Corrected local AU-PRO

New localisation outputs identify `FPR-boundary-left-limit-v1`. `au_pro_30` integrates the checked curve through normal-pixel FPR0.30; a vertical segment at the cutoff has zero area. `au_pro_30_legacy` retains the historical interpolation calculation as a separate value. Both remain201-quantile local approximations, not publisher private scores.

The checked area accepts at most1e-8 of floating-point overshoot before clamping to[0,1]. This numerical allowance covers float64 histogram sums of inverse region sizes across large pixel regions; it does not modify the separately reported legacy value. Invalid/out-of-range curves still fail.

Existing report, protocol, selection, fitted model and checksum files are not rewritten. An old `au_pro_30` value without the new revision remains historical legacy evidence. It must not be relabelled as a corrected result. A genuinely new scientific execution needs a new versioned workspace and protocol. Source/dependency guards remain active; never update a frozen source hash merely to accept changed code.

## Completed development fits

New fits use `vision-fit-v2` result metadata with `result_hash`. Reusing a completed fit verifies its metadata fingerprint, exact declared file set and member hashes, then independently recomputes its image metrics from the verified prediction parquet. Resealing an inconsistent metric does not make it valid.

Completed legacy fits can be read without adding a seal or changing bytes when the original frozen selection pins `development.json` by SHA256 and the legacy result exactly matches its entries. This also anchors the original measured inference time, which cannot be derived from predictions. Scientific metrics are still independently checked against prediction data. A legacy fit without that existing development/selection anchor is rejected rather than silently upgraded. Historical scientific execution still requires its original frozen source; this read-only validation does not relax source guards.

## Publication

`verified_final_artifacts` is read-only. It verifies selection/protocol fingerprints, complete final checksums, the first-access model ledger and predictor configuration/seed/training identity metadata. `verify_final` retains the report-only interface.

A new serving bundle uses that verified selection snapshot and checks the actual copied predictor hashes against the snapshot's first-access pins before publication. A changed source file during copying cannot be accepted merely by producing new bundle checksums. Final report figures similarly consume a single prediction-byte snapshot matched to the verified checksum before the final public JSON is written. No verification path starts missing training or creates new scientific receipts.

Existing serving bundles continue to verify against their own unchanged manifests. Publication does not select a replacement model, reinterpret old localisation values or certify factory use.

## Optional AD2 storage routing

Ordinary imports and native local datasets do not require any private machine module. AD2 alone can opt into a reviewed local storage adapter by setting the process environment variable `SECOM_AD2_STORAGE_ADAPTER` to an absolute Python module path. The adapter must provide `resolve`, `member`, `member_root` and `names`; its source SHA256 is recorded in new AD2 protocols. Missing or changed configured adapters fail visibly. No native-path fallback silently ignores a configured adapter.

For the existing laptop's separately maintained dataset migration binding, a terminal can opt in without embedding a username or drive location in source:

```powershell
$env:SECOM_AD2_STORAGE_ADAPTER = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.dodo-dataset-storage/ad2_runtime_binding.py'
```

This chooses the existing local module; it does not move data or change its map/amendment/volume checks. Use a trusted reviewed module, since selecting it explicitly loads Python code. General application JSON/hash helpers remain native and independent of this AD2-only routing choice.
