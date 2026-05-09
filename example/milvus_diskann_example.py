from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

connections.connect("default", host="localhost", port="19530")

# Schema Definition
vec_field  = FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=128)
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
schema = CollectionSchema(fields=[id_field, vec_field])
col = Collection(name="diskann_example", schema=schema)

# Data Insersion
import numpy as np
xb = np.random.random((100, 128)).astype("float32").tolist()
ids = list(range(100))
col.insert([ids, xb])

# === Important: DiskANN Index Build ===
col.create_index(
    field_name="vec",
    index_params={"index_type": "DISKANN", "metric_type": "L2", "params": {}}
)

# Load for search
col.load()

# Search
xq = np.random.random((5, 128)).astype("float32").tolist()
res = col.search(data=xq, anns_field="vec", param={"search_list": 64}, limit=10, output_fields=[])
print(res[0].ids, res[0].distances)

col.release()
col.drop()


'''
Print example
[52, 3, 66, 45, 78, 20, 79, 42, 54, 90] [17.881370544433594, 17.978273391723633, 18.12938117980957, 18.254981994628906, 18.30390167236328, 18.340600967407227, 18.487356185913086, 18.521326065063477, 18.61063003540039, 18.654346466064453]
'''