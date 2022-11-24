# GRADE
An implementation for "Non-IID Transfer Learning on Graphs" (AAAI'23).

## Environment Requirements
The code has been tested under Python 3.6.5. The required packages are as follows:
* numpy==1.18.1
* torch==1.4.0
* torchvision==0.5.0

## Data Sets
We used the following data sets in our experiments:
* [Airport Networks](https://github.com/GentleZhu/EGI/tree/main/data)

## Run the Codes
For cross-network node classification, please run
```
python main.py
```

## Acknowledgement
This is the latest source code of **GRADE** for AAAI-2023. If you find that it is helpful for your research, please consider to cite our paper:

```
@inproceedings{wu2022noniid,
  title={Non-IID Transfer Learning on Graphs},
  author={Wu, Jun and He, Jingrui and Ainsworth, Elizabeth},
  booktitle={Thirty-Seventh AAAI Conference on Artificial Intelligence},
  year={2023}
}
```
