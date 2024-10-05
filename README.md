# EIEA
This repository is the official implementation of EIEA.


# Environment
The essential package and recommened version to run the code:
```
pip install -r EIEA-env.txt
```

# Dataset
The **MMEA-data** and **ECS-results** can be download from (GoogleDrive).
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

# Run EIEA
python runxxx.py DATASET RATE 

For example: 
```
python runxxx.py FB_YAGO 0.2
python runxxx.py FB_DB 0.2
python runxxx.py FB_DB 0.5
```
