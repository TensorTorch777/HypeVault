# Dataset integrity audit

Phase 47 audited the existing five-brand catalog. The protocol was hashed before the additional images were drawn. The locked final-test files were not opened. No model was trained, scored, or recalibrated, and no directory label was changed.

`DATASET_LABEL_INTEGRITY = FAIL`

The failure is an evaluation failure. The directory labels are internally consistent under the stored duplicate rule, and many inspected images do show a watch of the named brand. What the catalog does not support is a claim that those labels are independently verified authenticity judgments, or that a near-perfect score on this catalog measures watch authenticity.

## What the labels are

`split_manifest_v2.json` records brand as the parent directory and authentic/counterfeit as the top-level directory (`Label_0_Watches` = 0, `Label_1_Watches` = 1). The same header sets source, marketplace, and product id to unavailable, and sets `product_disjoint_guarantee` to false. Filenames are content hashes.

Independent provenance coverage in the audited sample is 0/55. The same absence applies to the catalog header for every row. A watermark visible in a photograph was recorded as an observation. It is not a stored source and it was not treated as verification.

## Audit sample

Protocol SHA-256 `db3fd5034850021f4841168e93c30fcdf7199af5698c6679843bb5a6a8b0d45c`.

Sample SHA-256 `e78ec64c75b1f2621638c15f8de5fc24ecf6d9bd8efbe62b58fe4c96a43f700b`.

The sample has 55 unique non-test images:

- 15 Richard Mille counterfeit images already identified in Phase 46
- 40 images from `random.Random(47)`, four per brand and four per label, pools sorted by SHA-256
- no overlap between those two strata
- no image from the locked test split

These 55 images are the visual sample. Rates below are counts inside that sample. They are not catalog rates, except where a later section says the figure is a non-test census.

| Finding in the 55 | Count |
| --- | ---: |
| Full watch as the framed subject | 39 |
| Dial crop | 11 |
| Movement or component | 4 |
| Collage or advertisement sheet | 1 |
| Suitable for a single-image watch task | 51 |
| Unsuitable | 2 |
| Suitability uncertain | 2 |
| Apparent brand consistent | 24 |
| Apparent brand inconsistent | 15 |
| Apparent brand uncertain | 16 |
| Independent provenance | 0 |
| Flagged for expert review | 31 |

Expert review is 31/55 of this sample (56%). The flag means a visible discrepancy or a third-party overlay. It does not mean the directory label was judged false.

All 20 stratified authentic images looked consistent with the folder brand. Six of them carry a third-party mark (`bestwatchaaa.com`, `DUKEJONES` plus a phone number, `EVANCE`, or `XV`) and are flagged because that overlay is the only source clue and it is not in the manifest.

The stratified counterfeit cell is mixed. Four of the 20 had a readable matching wordmark. Eight had broken or misspelled brand lettering. The rest did not show a readable brand name. One Audemars Piguet counterfeit image is a movement macro with no case, so it is unsuitable for a single-watch photograph task. One Richard Mille counterfeit image in the preidentified set is an advertisement sheet.

Per brand, inside this sample:

| Brand | n | Consistent | Inconsistent | Uncertain | Expert review |
| --- | ---: | ---: | ---: | ---: | ---: |
| A. Lange & Söhne | 8 | 6 | 2 | 0 | 3 |
| Audemars Piguet | 8 | 4 | 1 | 3 | 5 |
| Patek Philippe | 8 | 5 | 2 | 1 | 4 |
| Richard Mille | 23 | 5 | 7 | 11 | 15 |
| Vacheron Constantin | 8 | 4 | 3 | 1 | 4 |

Richard Mille is larger because the 15 preidentified errors were added on top of its stratified cell.

## Class preparation, measured on every non-test file

The non-test census is 26,994 images. Test files were not opened.

Every one of the 13,500 non-test counterfeit images is 512×512. Of the 13,494 non-test authentic images, one is 512×512: `79313cbd6863.jpg`, an A. Lange & Söhne validation dial graphic with no case. The other authentic files use 3,611 other shapes. About one third of authentic files are square, and their median aspect ratio is 1.0 only because many catalog photos happen to be square at other sizes.

A rule that calls 512×512 counterfeit and every other size authentic is correct on 26,993 of 26,994 non-test files. File-size medians follow the same split in every brand: authentic medians are about 153 KB to 241 KB, and counterfeit medians are about 98 KB to 109 KB. Outer-edge standard deviation is also higher for counterfeit files in every brand (medians about 61 to 72, against about 34 to 45 for authentic files).

The research loader resizes every image to 512 before the network, so raw width is not a numeric input. The two classes were still prepared differently. Resampling, JPEG compression, and padding can remain after that resize. This audit did not test whether the trained model uses those remnants. The measurement that stands on its own is that directory class is identifiable from file geometry before any watch is examined.

## Duplicates and leakage

Under the stored dHash groups:

- 0 groups mix authentic and counterfeit
- 0 groups cross train, calibration, validation, and test
- 1,799 multi-member groups contain 4,243 images, and every one of those groups is authentic-only
- 118 of those groups cross brand directories (495 authentic images) at Hamming 0 without sharing SHA-256

Two SHA-256 values are repeated. Both are gray images of a white “JS” monogram, with no watch. `134681818e00.jpg` is copied into four authentic brand folders. `2889a285fade.jpg` is copied into all five. All nine copies are in train, and both files share one dHash group. Brand labels on those copies conflict with each other and with the picture.

Source-family independence is not established. Nothing in the manifest records a seller, listing, or product id, so images from one listing can sit in different splits even though duplicate groups do not.

Five of the 55 audited images sit in a multi-member group. All five are authentic.

## Richard Mille errors

Phase 45 recorded 20 false-authentic Richard Mille counterfeits. The saved fold artifacts list 15 of them. Those 15 identities were found again in the Phase 45 and Phase 46 files. The other five were not written anywhere those files store paths. Fold weights are gone, and this audit did not reconstruct the missing five with another checkpoint.

The 15 were not scored again. Saved P(fake) runs from 0.000274 to 0.075324, all below the frozen threshold 0.50.

| Category among the 15 | Count |
| --- | ---: |
| Full watch | 11 |
| Dial crop | 1 |
| Movement or component | 2 |
| Advertisement sheet | 1 |
| Apparent brand inconsistent | 7 |
| Apparent brand uncertain | 8 |
| Apparent brand consistent | 0 |
| Independent review flagged | 9 |

Six of the 15 are ordinary studio tonneau photographs with no recorded wordmark defect. Seven show a broken wordmark or a case shape that does not match the Richard Mille tonneau. Those notes are observations. They are not a claim that lettering or composition caused the low score.

## What Phase 45 still means

Phase 45’s recorded metrics are unchanged. `CROSS_BRAND_GENERALIZATION = STRONG` remains the result of the pre-specified rule inside this catalog: mean unseen PR-AUC 0.9999969178872116, mean unseen false-authentic rate 0.002074074074074074, mean PR-AUC gap 3.082112788499458e-06. Richard Mille was the weakest held-out brand.

That result is separation of the two directory classes across the five brands. It remains interpretable as that separation. It does not remain interpretable as evidence that the model learned independently verified authenticity, because:

- no image has a recorded source or expert label
- counterfeit files were uniformly prepared at 512×512 and authentic files were not
- identical non-watch placeholders were filed as authentic under several brand names

`PHASE45_GENERALIZATION_SCOPE = FIVE_BRAND_CATALOG_ONLY`

`EXTERNAL_SIXTH_BRAND_VALIDATION = BLOCKED`

## Evidence required before more model development

A later training phase needs a corpus in which each of the following is true before any new fit:

- authentic and counterfeit labels come from a recorded source that is independent of the directory name and independent of the model
- brand identity is recorded separately from the folder name, with conflicts such as the JS placeholders removed or quarantined
- both classes are stored at their original geometry, or both are resized by one documented transform that is applied after the split is frozen
- source or listing ids exist, and the split is disjoint in those ids
- the locked final test stays sealed until that corpus and its protocol are hashed

Until those conditions exist, another training run on this catalog would fit the same directory convention.
