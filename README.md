# DagoBERT

This repository contains the code and data for the EMNLP paper [DagoBERT: Generating Derivational Morphology
with a Pretrained Language Model](https://www.aclweb.org/anthology/2020.emnlp-main.316.pdf). 
The paper introduces **DagoBERT (Derivationally and generatively optimized BERT)**, a BERT-based model for generating 
derivationally complex words.

# Dependencies

The code requires Python (>=3.6), NumPy (>=1.18), PyTorch (>=1.2), and Transformers (>=2.5).

# Data

# Usage

To replicate the experiment on the best segmentation method, run the script test_segmentation.sh in src/model/.
No training is required for this experiment since pretrained BERT is used.

To replicate the main experiment, run the script train_main.sh in src/model/.
After training has finished, run the script test_main.sh in src/model/.

To replicate the experiment on the Vylomova et al. (2017) dataset, run the script train_vyl.sh in src/model/.
After training has finished, run the script test_vyl.sh in src/model/.

To replicate the experiment on the impact of the input segmentation, run the script train_mwf.sh in src/model/.
After training has finished, run the script test_mwf.sh in src/model/.

The scripts expect the full dataset in data/final/.

# Citation

If you use the code or data in this repository, please cite the following paper:

```
@inproceedings{hofmann2020dagobert,
    title = {Dago{BERT}: Generating Derivational Morphology with a Pretrained Language Model},
    author = {Hofmann, Valentin and Pierrehumbert, Janet and Sch{\"u}tze, Hinrich},
    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
    year = {2020}
}

```
