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
Thanks to Deng Weijian for providing the code for *[generating sample sets](https://github.com/Simon4Yan/Meta-set)*. Please refer to *https://github.com/Simon4Yan/Meta-set*, to generated datasets to train regression model.

*****


### Run the Code

 1. Creat sample sets and 2. Train classifier and get image features of sample sets
     
     pleaser refer 
    *https://github.com/Simon4Yan/Meta-set/blob/main/meta_set*

 3. Get set representations
    ```bash
    # get shape, clusters and sampled data.  
    python Set_rep/get_set_representation.py
    ```

 4. Get set representations
    ```bash
    # get shape, clusters and sampled data.  
    python Set_rep/train_regnet_new.py
    ```








