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

To reproduce the results of SCAN, you can sequentially run the three `.sh` files:

    ```
    sh scripts/DAE.sh
    sh scripts/beta_VAE.sh
    sh scripts/SCAN.sh
    ```

By default, the working environment setting is `/data/SCAN/` in `main.py` `--root_dir`, which is overriden and set as `/data/hc/SCAN` by the three `.sh` files above.
This is directory suppose to contain the dataset `root_dir/dataset/`, and is where checkpoint files and output files will be saved.
The original beta-VAE models are still supported, and examples of result reproducing commands can be found in `scripts/original-beta_VAE/`
To view the original beta-VAE model, you can also refer to the parent repo [Github Repo][]


### Results

The convergence of SCAN is not salient in plots of the losses, compared to more obvious convergence of DAE and beta-VAE models.
[reconstruction loss](results/SCAN/recon.png)

Pretrained DAE and beta-VAE results can be found in `pretrained_results.md`.

### Acknowledgement:

I've referred to [this issue](https://github.com/miyosuda/scan/issues/1), and adopted its solution which is to use the DAE output rather than to improve the visuality of beta-VAE.

### Reference
1. [SCAN: Learning Hierarchical Compositional Visual Concepts, Higgins et al., ICLR 2018]
2. [Github Repo]: Pytorch implementation of beta-VAE from [1Konny](https://github.com/1Konny)

[SCAN: Learning Hierarchical Compositional Visual Concepts, Higgins et al., ICLR 2018]: https://arxiv.org/abs/1707.03389
[Github Repo]: https://github.com/1Konny/Beta-VAE 
[same with here]: https://github.com/1Konny/FactorVAE 
