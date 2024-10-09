# EIEA
This repository is the official implementation of EIEA (Explicit-Implicit Entity Alignment Method in Multi-modal Knowledge Graphs).


# Environment
The essential package and recommened version to run the code:
```
pip install -r EIEA-env.txt
```

# Dataset
The **MMEA-data** and **ECS-results** can be download from  [GoogleDrive](https://drive.google.com/drive/folders/1wfErYdAV93yxPtPHqkGanbmb_Ztv-LRU?usp=drive_link).
The original MMEA dataset can be download from  [MMKB](https://github.com/mniepert/mmkb) and [MMEA](https://github.com/lzxlin/mclea?tab=readme-ov-file).
Those files should be organized into the following file hierarchy:
```
EIEA
├── data
│   └── ECS_results
|        └── seed0.2
|              └── FB15K-DB15K
|              └── FB15K-YAGO15K
|        └── seed0.5
|        └── seed0.8
|   └── MMEA_name
|   └── MMEA-data
|        └── seed0.2
|              └── FB15K-DB15K
|              └── FB15K-YAGO15K
|        └── seed0.5
|        └── seed0.8
└── src
└── ECS_compute

```
**NOTE**
Based on the URLs given in the original dataset, we use crawling techniques to obtain entity name information. However, due to the unavailability of some URLs, the name information of certain entities in FB15K is missing. 
All entity name information can be found in **MMEA_name**.

# Run EIEA
python runxxx.py DATASET RATE 

For example: 
```
python run0.2.py FB_YAGO 0.2
python run0.2.py FB_DB 0.2
python run0.2.py FB_DB 0.5
```
# EIEA on FB-DB/ FB-YAGO

| Models         | FB15K-DB15K 20% seed |        |        | FB15K-DB15K 50% seed |        |        | FB15K-DB15K 80% seed |        |        | FB15K-YG15K 20% seed |        |        | FB15K-YG15K 50% seed |        |        | FB15K-YG15K 80% seed |        |        |
|----------------|----------------------|--------|--------|----------------------|--------|--------|----------------------|--------|--------|----------------------|--------|--------|----------------------|--------|--------|----------------------|--------|--------|
|                | H@1                  | H@10   | MRR    | H@1                  | H@10   | MRR    | H@1                  | H@10   | MRR    | H@1                  | H@10   | MRR    | H@1                  | H@10   | MRR    | H@1                  | H@10   | MRR    |
| PoE            | 12.6                 | 25.1   | 17.0   | 46.4                 | 65.8   | 53.3   | 66.6                 | 82.0   | 72.1   | 11.3                 | 22.9   | 15.4   | 34.7                 | 53.6   | 41.4   | 57.3                 | 74.6   | 63.5   |
| MMEA           | 26.4                 | 54.1   | 35.7   | 41.7                 | 70.3   | 51.2   | 59.0                 | 86.9   | 68.5   | 23.4                 | 48.0   | 31.7   | 40.3                 | 64.5   | 48.6   | 59.8                 | 83.9   | 68.2   |
| EVA            | 55.5                 | 71.5   | 35.7   | -                    | -      | -      | -                    | -      | -      | 10.2                 | 27.7   | 16.4   | 41.5                 | 60.3   | 48.5   | 53.7                 | 81.0   | 64.9   |
| MCLEA          | 44.5                 | 70.5   | 53.4   | 57.3                 | 80.0   | 65.2   | 73.0                 | 88.3   | 78.4   | 38.8                 | 64.1   | 47.4   | 57.4                 | 78.4   | 65.1   | 65.3                 | 83.5   | 71.5   |
| MSNEA          | 65.2                 | 81.1   | 70.8   | -                    | -      | -      | -                    | -      | -      | 44.2                 | 69.2   | 57.2   | 54.3                 | 75.9   | 61.6   | 65.3                 | 83.5   | 71.5   |
| MEAformer      | 57.8                 | 81.1   | 68.8   | 71.8                 | 91.2   | 79.4   | 92.1                 | 98.4   | 92.1   | 43.0                 | 61.2   | 51.2   | 47.8                 | 79.9   | 61.2   | 77.1                 | 88.7   | 82.7   |
| DESAlign       | 58.0                 | 81.1   | 65.5   | 71.8                 | 89.0   | 78.9   | 92.4                 | 98.4   | 92.1   | 47.1                 | 63.4   | 55.0   | 50.2                 | 77.7   | 61.7   | 77.0                 | 88.7   | 82.6   |
| XGEA           | 68.2                 | 84.8   | 72.9   | 79.9                 | 95.0   | 85.6   | -                    | -      | -      | 48.3                 | 64.4   | 57.9   | 61.2                 | 79.3   | 67.0   | 77.5                 | 94.6   | 84.7   |
| PCMEA          | 66.7                 | 84.8   | 72.8   | 73.7                 | 91.5   | 78.1   | 96.4                 | 100.0  | 85.8   | 51.4                 | 67.8   | 60.4   | 63.8                 | 85.8   | 72.1   | 75.5                 | 94.6   | 84.7   |
| ASGEA          | 62.8                 | 79.9   | 68.9   | 73.9                 | 89.9   | 85.2   | 92.7                 | 100.0  | 88.1   | 71.7                 | 84.8   | 77.6   | 71.9                 | 88.4   | 77.0   | 91.5                 | 98.8   | 91.1   |
| EIEA w/o N     | 74.2                 | 85.3   | 78.1   | 86.1                 | 94.5   | 89.2   | 94.5                 | 98.4   | 94.4   | 83.6                 | 94.1   | 86.9   | 86.7                 | 94.9   | 91.5   | 94.9                 | 99.8   | 98.9   |
| EIEA           | 92.2                 | 97.7   | 94.7   | 97.1                 | 99.4   | 98.0   | 99.7                 | 100.0  | 100.0  | 96.0                 | 97.7   | 97.0   | 98.2                 | 98.8   | 99.8   | 99.8                 | 100.0  | 99.8   |
| EIEA w/ CSP    | 92.8                 | 97.9   | 94.7   | 97.2                 | 99.3   | 98.0   | 99.7                 | 100.0  | 100.0  | 96.3                 | 98.8   | 97.2   | 97.6                 | 99.4   | 98.2   | 99.1                 | 99.9   | 99.1   |

# EIEA On DBP15K
| DBP15K  | Ja_en          |              |                |Fr_en         |                |              |Zh_en         |                |              |  
|---------|----------------|--------------|----------------|--------------|----------------|--------------|--------------|----------------|--------------|
|         | H@1            | H@10         | MRR            | H@1          | H@10           | MRR          | H@1          | H@10           | MRR          |
| EIEA    | **98.00**      | **99.72**    | **98.69**      | **99.60**    | **99.98**      | **99.75**    | **96.14**    | **99.35**      | **97.37**    |




