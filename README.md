# SCAN
Pytorch reproduction of the paper:
[SCAN: Learning Hierarchical Compositional Visual Concepts, Higgins et al., ICLR 2018]

### Dependencies
```
python 3.6.4
pytorch 0.3.1.post2
visdom
```

### Datasets
[same with here]

### Usage
To initialize visdom:

    ```
    vidsom -port 6059
    ```

To reproduce the results of SCAN, sequentially run the three commands:

    ```
    sh scripts/DAE.sh
    sh scripts/beta_VAE.sh
    sh scripts/SCAN.sh
    ```

The original beta-VAE models are still supported, and examples of commands reproducing the results can be found in `scripts/original-beta_VAE/`

To view them, you can also refer to the parent repo [Github Repo][]

Acknowledgement:

    I've referred to [this issue](https://github.com/miyosuda/scan/issues/1), and used the DAE output to improve the visuality of beta-VAE.

### Reference
1. [SCAN: Learning Hierarchical Compositional Visual Concepts, Higgins et al., ICLR 2018]
2. [Github Repo]: Pytorch implementation of beta-VAE from [1Konny](https://github.com/1Konny)

[SCAN: Learning Hierarchical Compositional Visual Concepts, Higgins et al., ICLR 2018]: https://arxiv.org/abs/1707.03389
[Github Repo]: https://github.com/1Konny/Beta-VAE 
[same with here]: https://github.com/1Konny/FactorVAE 
