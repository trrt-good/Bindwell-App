# Validation
We compared the performance of our model, AffinityLM, against several other industry methods, including [CAPLA, CAPLA-Pred](https://doi.org/10.1093/bioinformatics/btad049), 
[DeepDTA](https://doi.org/10.1093/bioinformatics/bty593), [DeepDTAF](https://doi.org/10.1093/bib/bbab072), [Pafnucy](https://doi.org/10.1093/bioinformatics/bty374), [OnionNet](https://doi.org/10.1021/acsomega.9b01997), [FAST](https://doi.org/10.1021/acs.jcim.0c01306),[ IMCP-SF](https://doi.org/10.1016/j.csbj.2022.02.004), [GLI](https://doi.org/10.1109/ICDM54844.2022.00175), and the affinity prediction model made by [Blanchard et al.](https://doi.org/10.1177/10943420221121804), on benchmarks described in Section 2.3.

**Table 1:** AffinityLM's performance on the Test-2016_290 dataset by [Jin et al.](https://doi.org/10.1093/bioinformatics/btad049). Higher R is better, lower RMSE and MAE are better. Best values are in bold.
| Method | R | RMSE | MAE |
| --- | --- | --- | --- |
| AffinityLM | **0.845** | **1.196** | **0.906** |
| CAPLA | 0.825 | 1.298 | 1.014 |
| CAPLA-Pred | 0.825 | 1.298 | 1.014 |
| DeepDTAF | 0.789 | 1.355 | 1.073 |
| Pafnucy | 0.775 | 1.418 | 1.129 |
| OnionNet | 0.816 | 1.278 | - |
| FAST | 0.810 | 1.308 | 1.019 |
| IMCP-SF | 0.791 | 1.452 | 1.155 |
| GLI | - | 1.294 | 1.026 |

**Table 2:** AffinityLM's performance on the CSAR-HiQ_36 dataset from the [Jin et al.](https://doi.org/10.1093/bioinformatics/btad049) paper. Higher R is better, lower RMSE and MAE are better. Best values are in bold.

| Method | R | RMSE | MAE |
| --- | --- | --- | --- |
| AffinityLM | 0.742 | **1.342** | **1.150** |
| Blanchard et al. | **0.774** | 1.484 | 1.176 |
| CAPLA | 0.704 | 1.454 | 1.160 |
| DeepDTAF | 0.543 | 2.765 | 2.318 |
| Pafnucy | 0.566 | 1.658 | 1.291 |
| IGN | 0.528 | 1.795 | 1.431 |
| IMCP-SF | 0.631 | 1.560 | 1.205 |
