# Dataset structure audit

Verified metadata:

- Label: top-level directory Label_0_Watches (authentic, 0) or Label_1_Watches (fake, 1)
- Brand: immediate parent directory name
- Filename: 12-character hexadecimal stem plus .jpg; not documented as a SKU

Unavailable metadata:

- product_id is UNKNOWN on every record.
- marketplace is UNKNOWN on every record.
- source is UNKNOWN on every record.
- No physical-watch or SKU identity is stored.

Current group: load_records sets split_group and product_group_key to the resolved parent directory. Each such directory is one brand inside one label folder. It is a brand-and-label bucket of many files, not a product or SKU.

True product-disjoint splitting is possible: false.
No product or SKU field exists. A parent directory holds thousands of images, and nothing on disk proves which of those images are the same physical watch.

Samples: 30000. Brands: 5. Current groups: 10.
Images per current group: 3000 to 3000.

Exact duplicate groups: 2. Largest exact group: 5.
Near-duplicate groups (dhash, Hamming <= 0): 1799. Largest near-duplicate group: 23.
Hamming 0 keeps identical dHash values together. Hamming 1 was rejected because its transitive component contains 826 authentic images from all five brands with pairwise median distance 6, which is not an obvious copy.
Near-duplicate groups are not product groups.
Manual-review images (mixed authentic/fake duplicate group): 0.

Corrected strategy: brand_stratified_duplicate_safe. product_disjoint_guarantee: false.
Membership hash: 78a38ebac1102977ec5598577126ade77fed8edd6563e13dbdee73dba389f678.

| Split | Total | Authentic | Fake | Brands |
| --- | ---: | ---: | ---: | ---: |
| train | 20999 | 10499 | 10500 | 5 |
| calibration | 2998 | 1498 | 1500 | 5 |
| validation | 2997 | 1497 | 1500 | 5 |
| test | 3006 | 1506 | 1500 | 5 |

| Split | Brand | Authentic | Fake | Total |
| --- | --- | ---: | ---: | ---: |
| train | A. Lange & Söhne | 2101 | 2100 | 4201 |
| train | Audemars Piguet | 2095 | 2100 | 4195 |
| train | Patek Philippe | 2095 | 2100 | 4195 |
| train | Richard Mille | 2102 | 2100 | 4202 |
| train | Vacheron Constantin | 2106 | 2100 | 4206 |
| calibration | A. Lange & Söhne | 301 | 300 | 601 |
| calibration | Audemars Piguet | 299 | 300 | 599 |
| calibration | Patek Philippe | 296 | 300 | 596 |
| calibration | Richard Mille | 300 | 300 | 600 |
| calibration | Vacheron Constantin | 302 | 300 | 602 |
| validation | A. Lange & Söhne | 315 | 300 | 615 |
| validation | Audemars Piguet | 297 | 300 | 597 |
| validation | Patek Philippe | 293 | 300 | 593 |
| validation | Richard Mille | 300 | 300 | 600 |
| validation | Vacheron Constantin | 292 | 300 | 592 |
| test | A. Lange & Söhne | 283 | 300 | 583 |
| test | Audemars Piguet | 309 | 300 | 609 |
| test | Patek Philippe | 316 | 300 | 616 |
| test | Richard Mille | 298 | 300 | 598 |
| test | Vacheron Constantin | 300 | 300 | 600 |
