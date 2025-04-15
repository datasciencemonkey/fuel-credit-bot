from databricks_langchain import DatabricksVectorSearch
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
import os
from rich import print

load_dotenv()

vector_store = DatabricksVectorSearch(
    endpoint='sg-endpoint',
    index_name='main.sgfs.fuel_card_terms',
    workspace_client=WorkspaceClient(
        host=os.getenv("DATABRICKS_HOST"),
        token=os.getenv("DATABRICKS_TOKEN"),
    ),
)


retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print(retriever.invoke("What is the late fees policy?"))