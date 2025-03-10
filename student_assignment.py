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
    pass


def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass


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
    generate_hw01()
