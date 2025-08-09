# ICLR2025 - Adaptive Retention & Correction: Test-Time Training for Continual Learning

 :bookmark_tabs:[`Paper Link`](https://arxiv.org/abs/2405.14318) **Authors**: [Haoran Chen](https://haoranchen.github.io/), [Micah Goldblum](https://goldblum.github.io/), [Zuxuan Wu](https://zxwu.azurewebsites.net/),  [Yu-Gang Jiang](https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=en)

## How to use

### Dependencies

1. [torch 2.0.1](https://github.com/pytorch/pytorch)
2. [torchvision 0.15.2](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models) (Note, for reproducing ARC + DER, we recommend using [timm 0.5.4](https://github.com/huggingface/pytorch-image-models))
4. [tqdm](https://github.com/tqdm/tqdm)
5. [numpy](https://github.com/numpy/numpy)
6. [scipy](https://github.com/scipy/scipy)


### Run experiment

1. Edit the `[MODEL NAME].json` file for global settings and hyperparameters.
2. Run:

    ```bash
    python main.py --config=./exps/[MODEL NAME].json
    ```
## Acknowledgments

We thank the [PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT) repo for providing helpful codebase in our work.

## Contact
Feel free to contact us if you have any questions or suggestions 
Email: chenhran21@m.fudan.edu.cn

## Citation
If you use our code in this repo or find our work helpful, please consider giving a citation:

```
@inproceedings{chenarc,
  title={Adaptive Retention \& Correction: Test-Time Training for Continual Learning},
  author={Chen, Haoran and Goldblum, Micah and Wu, Zuxuan and Jiang, Yu-Gang},
  booktitle={ICLR},
  year={2025}
}
```
