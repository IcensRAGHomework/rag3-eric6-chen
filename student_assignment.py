import datetime
import chromadb
import traceback
import csv

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = "text-embedding-ada-002"
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

csv_file_path = r"COA_OpenData.csv"


def date_to_epoch(date_string):
    return datetime.datetime.strptime(date_string, "%Y-%m-%d").timestamp()


def generate_hw01():
    # copy paste from demo()
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config["api_key"],
        api_base=gpt_emb_config["api_base"],
        api_type=gpt_emb_config["openai_type"],
        api_version=gpt_emb_config["api_version"],
        deployment_id=gpt_emb_config["deployment_name"],
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL", metadata={"hnsw:space": "cosine"}, embedding_function=openai_ef
    )
    if collection.count != 0:
        # avoid data duplication
        print("already data in DB")
        return collection
    # read from CSV
    with open(csv_file_path, encoding="utf8") as file:
        print("file opened")
        reader = csv.DictReader(file, delimiter=r",")
        print("csv reader created")
        index = 0
        for row in reader:
            metadata = {
                "file_name": csv_file_path,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": date_to_epoch(row["CreateDate"]),
            }
            collection.add(
                ids=[str(index)], metadatas=[metadata], documents=[row["HostWords"]]
            )
            index += 1

    print("method finished")
    return collection


def generate_hw02(question, city, store_type, start_date, end_date):
    similarity_threshold = 0.80
    distance_threshold = 1 - similarity_threshold
    n_results = 10
    collection = generate_hw01()
    query = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["metadatas", "distances"],
        where={
            "$and": [
                {"date": {"$gte": int(start_date.timestamp())}},
                {"date": {"$lte": int(end_date.timestamp())}},
                {"type": {"$in": store_type}},
                {"city": {"$in": city}},
            ]
        },
    )
    # list comprehension
    names = [
        metadata["name"]
        for metadata, distance in zip(query["metadatas"][0], query["distances"][0])
        if distance < distance_threshold
    ]
    return names


def generate_hw03(question, store_name, new_store_name, city, store_type):
    # use existing data from hw01
    collection = generate_hw01()
    udpate_target = collection.get(where={"name": store_name})
    # print(udpate_target["metadatas"][0])
    for a_metadata in udpate_target["metadatas"]:
        a_metadata["new_store_name"] = new_store_name
    collection.upsert(
        ids=udpate_target.get("ids", []),
        metadatas=udpate_target["metadatas"],
        documents=udpate_target.get("documents", []),
    )

    similarity_threshold = 0.80
    distance_threshold = 1 - similarity_threshold
    n_results = 10
    query = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["metadatas", "distances"],
        where={
            "$and": [
                {"type": {"$in": store_type}},
                {"city": {"$in": city}},
            ]
        },
    )
    # seems query is already sorted so i decide to ignore sorting.
    names = [
        metadata.get(
            "new_store_name",
            # abuse default value parameter, assume "name" always exists.
            metadata["name"],
        )
        for metadata, distance in zip(query["metadatas"][0], query["distances"][0])
        if distance < distance_threshold
    ]
    return names


def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config["api_key"],
        api_base=gpt_emb_config["api_base"],
        api_type=gpt_emb_config["openai_type"],
        api_version=gpt_emb_config["api_version"],
        deployment_id=gpt_emb_config["deployment_name"],
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL", metadata={"hnsw:space": "cosine"}, embedding_function=openai_ef
    )

    return collection


if __name__ == "__main__":
    # print("test")
    # print(demo("test"))
    # generate_hw01()
    # 02: sample from readme
    # print(
    #     generate_hw02(
    #         "我想要找有關茶餐點的店家",
    #         ["宜蘭縣", "新北市"],
    #         ["美食"],
    #         datetime.datetime(2024, 4, 1),
    #         datetime.datetime(2024, 5, 1),
    #     )
    # )

    # 03: sample from readme
    print(
        generate_hw03(
            "我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵",
            "耄饕客棧",
            "田媽媽（耄饕客棧）",
            ["南投縣"],
            ["美食"],
        )
    )
