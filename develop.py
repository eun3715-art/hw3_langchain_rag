"""
develop.py - 벡터 DB 초기 셋업 스크립트
실행: python develop.py

1단계: Milvus 컬렉션 생성 및 데이터 삽입
2단계: LangChain 인터페이스로 검색 동작 확인
"""

import pickle
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient, DataType
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from engine.embeddings import CLIPEmbeddings
from engine.vectorstore import MilvusVectorStore

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
VECTOR_DB_DIR.mkdir(exist_ok=True)

COLLECTION_NAME_TEXT = "text_search"
COLLECTION_NAME_IMAGE = "image_search"

# --- 1단계: Milvus DB 셋업 ---

print("Milvus Lite 연결 중...")
client = MilvusClient(uri="./milvus_local.db")
print("연결 완료")

print("임베딩 데이터 로드 중...")
txt_embeddings = np.load(DATA_ROOT / "txt_embeddings.npy").astype("float32")
img_embeddings = np.load(DATA_ROOT / "img_embeddings.npy").astype("float32")
print(f"텍스트 임베딩: {txt_embeddings.shape}, 이미지 임베딩: {img_embeddings.shape}")

text_dir = DATA_ROOT / "texts"
text_contents = {}
text_ids = []
text_vectors = []

for txt_file in sorted(text_dir.glob("*.txt")):
    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    text_id = int(txt_file.stem)
    text_contents[text_id] = content
    text_ids.append(text_id)
    text_vectors.append(txt_embeddings[text_id])

print(f"텍스트 파일 로드: {len(text_contents)}개")

for name in [COLLECTION_NAME_TEXT, COLLECTION_NAME_IMAGE]:
    try:
        client.drop_collection(collection_name=name)
    except Exception:
        pass

print("텍스트 컬렉션 생성 중...")
schema_text = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema_text.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema_text.add_field(field_name="text_vector", datatype=DataType.FLOAT_VECTOR, dim=512)

index_params = client.prepare_index_params()
index_params.add_index(field_name="text_vector", metric_type="IP", index_type="FLAT")

client.create_collection(
    collection_name=COLLECTION_NAME_TEXT,
    schema=schema_text,
    index_params=index_params
)

print("텍스트 데이터 삽입 중...")
insert_data = []
for text_id, content in text_contents.items():
    idx = text_ids.index(text_id)
    insert_data.append({
        "id": text_id,
        "text_vector": text_vectors[idx].tolist(),
        "text_content": content
    })

batch_size = 500
for i in range(0, len(insert_data), batch_size):
    batch = insert_data[i:i + batch_size]
    client.insert(collection_name=COLLECTION_NAME_TEXT, data=batch)
    print(f"  {min(i + batch_size, len(insert_data))}/{len(insert_data)} 삽입")

print("이미지 컬렉션 생성 중...")
schema_image = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema_image.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema_image.add_field(field_name="image_vector", datatype=DataType.FLOAT_VECTOR, dim=512)

index_params_img = client.prepare_index_params()
index_params_img.add_index(field_name="image_vector", metric_type="IP", index_type="FLAT")

client.create_collection(
    collection_name=COLLECTION_NAME_IMAGE,
    schema=schema_image,
    index_params=index_params_img
)

print("이미지 데이터 삽입 중...")
img_insert_data = [
    {"id": img_id, "image_vector": img_embeddings[img_id].tolist()}
    for img_id in range(len(img_embeddings))
]

for i in range(0, len(img_insert_data), batch_size):
    batch = img_insert_data[i:i + batch_size]
    client.insert(collection_name=COLLECTION_NAME_IMAGE, data=batch)
    print(f"  {min(i + batch_size, len(img_insert_data))}/{len(img_insert_data)} 삽입")

print("검색 동작 확인...")
results = client.search(
    collection_name=COLLECTION_NAME_TEXT,
    data=[txt_embeddings[0].tolist()],
    limit=3,
    output_fields=["text_content"]
)
print("텍스트 검색 결과 (top-3):")
for i, r in enumerate(results[0], 1):
    print(f"  {i}. ID={r['id']} score={r['distance']:.4f} | {r['entity']['text_content'][:60]}...")

metadata = {
    "text_contents": text_contents,
    "text_embeddings_shape": txt_embeddings.shape,
    "image_embeddings_shape": img_embeddings.shape,
    "collection_name_text": COLLECTION_NAME_TEXT,
    "collection_name_image": COLLECTION_NAME_IMAGE,
}
with open(VECTOR_DB_DIR / "metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("1단계 완료: 벡터 DB 셋업 완료")


# --- 2단계: LangChain 인터페이스 확인 ---

print("\n2단계: LangChain 인터페이스 확인")

print("CLIP 모델 로드 중...")
clip_embeddings = CLIPEmbeddings()

text_vectorstore = MilvusVectorStore(client=client, collection_name="text_search", embeddings=clip_embeddings)
image_vectorstore = MilvusVectorStore(client=client, collection_name="image_search", embeddings=clip_embeddings)

test_query = "A man is cooking in the kitchen"
print(f"\n텍스트 검색: '{test_query}'")
text_docs = text_vectorstore.similarity_search_with_score(test_query, k=3)
for i, (doc, score) in enumerate(text_docs, 1):
    print(f"  {i}. score={score:.4f} | {doc.page_content[:80]}...")

print(f"\n이미지 검색 (텍스트 쿼리): '{test_query}'")
image_docs = image_vectorstore.similarity_search_with_score(test_query, k=3)
for i, (doc, score) in enumerate(image_docs, 1):
    print(f"  {i}. {doc.page_content} | score={score:.4f}")

search_chain = RunnableParallel(
    text_results=RunnablePassthrough() | (lambda x: text_vectorstore.similarity_search(x, k=3)),
    image_results=RunnablePassthrough() | (lambda x: image_vectorstore.similarity_search(x, k=3))
)

chain_query = "People eating together"
print(f"\n병렬 체인 테스트: '{chain_query}'")
try:
    chain_results = search_chain.invoke(chain_query)
    print(f"  텍스트 결과: {len(chain_results['text_results'])}개")
    print(f"  이미지 결과: {len(chain_results['image_results'])}개")
except Exception as e:
    print(f"  체인 테스트 실패: {e}")

print("\n2단계 완료")
