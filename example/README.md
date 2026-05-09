# Example

Run the sample scripts in the example/ folder with a minimal Milvus setup.


## Abstract

Quick start to execute example/ code: create a Conda env, install deps, start Milvus, and run milvus_diskann_example.py.


## Quickstart 

### For MAC
```
# 1) Create and activate environment
conda create --name milvus python=3.11 -y
conda activate milvus

# 2) Install dependencies
pip install "pymilvus==2.4.9" "marshmallow>=3.13,<4" "environs>=9.5,<12" "grpcio>=1.59,<2"

# 3) Start Milvus
bash start.sh

# 4) Run the example (from the example/ folder)
python milvus_diskann_example.py
```

### For Windows
```
# 1) Create and activate environment
conda create --name milvus python=3.11 -y
conda activate milvus

# 2) Install dependencies
pip install "pymilvus==2.4.9" "marshmallow>=3.13,<4" "environs>=9.5,<12" "grpcio>=1.59,<2"

# 3) Start Milvus
mkdir milvus
docker run -d --name milvus-standalone --security-opt seccomp:unconfined -e ETCD_USE_EMBED=true -e ETCD_DATA_DIR=/var/lib/milvus/etcd -e COMMON_STORAGETYPE=local -e DEPLOY_MODE=STANDALONE -v %cd%\milvus:/var/lib/milvus -p 19530:19530 -p 9091:9091 -p 2379:2379 --health-cmd="curl -f http://localhost:9091/healthz" --health-interval=30s --health-start-period=90s --health-timeout=20s --health-retries=3 milvusdb/milvus:v2.4.9 milvus run standalone

# 4) Run the example (from the example/ folder)
python milvus_diskann_example.py
```


Scope: This setup is exclusively for running the example code under example/.


Tips  
- If bash start.sh starts services in the background, give them a moment to become ready before step 4.  