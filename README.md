## Label-Free Model Evaluation with Semi-Structured Dataset Representations

![fig1](https://github.com/sxzrt/Semi-Structured-Dataset-Representations/blob/main/imgs/fig-sys.jpg)  


### Prerequisites
This code uses the following libraries

- Python 3.7
- NumPy
- PyTorch 1.7.0 + torchivision 0.8.1
- Sklearn
- Scipy 1.2.1
*****

### Data Preparation
Thanks to Deng Weijian for providing the code for *[generating sample sets](https://github.com/Simon4Yan/Meta-set)*. Please refer to *https://github.com/Simon4Yan/Meta-set*, to generated datasets to train regression model. The newly collected datasets are avalibale at [link1](https://drive.google.com/file/d/1w6Df16ZbKKstHEhKjz8Ao4jf3vg-02aF/view?usp=sharing) and [link2](https://drive.google.com/file/d/18QlKo14uF_lMGeHW8C9X44luazE3GSIH/view?usp=sharing).

*****


### Run the Code

 1. Creat sample sets and 2. Train classifier and get image features of sample sets
     
     pleaser refer to
     
    *https://github.com/Simon4Yan/Meta-set/blob/main/meta_set*

 3. Get set representations
    ```bash
    # get shape, clusters and sampled data.  
    python Set_rep/get_set_representation.py
    ```

 4. Train the regresssor on dataset representaions
    ```bash 
    python Set_rep/train_regnet_new.py
    ```
    
 5. Test the regresssor 
    ```bash
    python Set_rep/test_regnet_new.py
    ```
    
****
### Citation
If you use the code in your research, please cite:

```bibtex
@article{DBLP:journals/corr/abs-2108-10310,
  author    = {Xiaoxiao Sun and
               Yunzhong Hou and
               Hongdong Li and
               Liang Zheng},
  title     = {Label-Free Model Evaluation with Semi-Structured Dataset Representations },
  journal   = {CoRR},
  volume    = {abs/2108.10310},
    url       = {https://arxiv.org/abs/2108.10310}
  year      = {2021},
}
```

```bibtex
@inproceedings{deng2020labels,
author={Deng, Weijian and Zheng, Liang},
title     = {Are Labels Always Necessary for Classifier Accuracy Evaluation?},
booktitle = {Proc. CVPR},
year      = {2021},
}
```





