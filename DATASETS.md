# Dataset access and splits

| Dataset | Access | Training / test centres |
| --- | --- | ---: |
| Houston2013 | [University of Houston dataset page](https://hyperspectral.ee.uh.edu/?page_id=459), [2013 contest paper](https://doi.org/10.1109/JSTARS.2014.2305441), [third-party prepared MAT files linked by S2ENet](https://drive.google.com/file/d/1cyA7sKQlh2c7qrIb7gzexivoyXG8Vie2/view?usp=sharing) | 2,832 / 12,197 |
| Trento | [Dataset reference](https://doi.org/10.1080/19479832.2015.1055833), [third-party download linked by HLMamba](https://drive.google.com/drive/folders/1HK3eL3loI4Wd-RFr1psLLmVLTVDLctGd) | 819 / 29,395 |
| MUUFL Gulfport | [Dataset and scene-label repository](https://github.com/GatorSense/MUUFLGulfport), [Zenodo archive](https://doi.org/10.5281/zenodo.1186326) | 550 / 53,137 |

Use the supplied training and test label masks for Houston2013 and Trento.
The expected scene grids are 349 rows by 1,905 columns for Houston2013 and
166 rows by 600 columns for Trento. File names may differ by distributor.

## Fixed MUUFL partition

Use the Campus 1 scene labels on the 325-row by 220-column grid. The file
[muufl_train_centres.csv](muufl_train_centres.csv) lists 550 zero-based
`row,col` pairs from the fixed experimental split, with 50 training centres
per class across 11 classes. The test set consists of all remaining labelled
pixels. Exclude unlabelled pixels from both sets. MATLAB readers should add
one to each coordinate before indexing.

The coordinates were reconstructed using the archived split-generation
procedure and the same label array input. The procedure applies a single
NumPy generator with split seed 1, visits class IDs in ascending order, and
takes the first 50 coordinates from each full class permutation. The split
seed determines the partition independently of the model training seed.

For MUUFL, cite Gader et al. (2013), *MUUFL Gulfport Hyperspectral and LiDAR
Airborne Data Set*, and Du and Zare (2017), *Scene Label Ground Truth Map for
MUUFL Gulfport Data Set*. Both references are listed by the dataset provider.
