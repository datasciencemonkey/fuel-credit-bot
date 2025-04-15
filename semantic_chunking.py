# %%
# Required imports
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import pandas as pd
from databricks.connect import DatabricksSession

# %%
# Set OpenAI API key
# os.environ["OPENAI_API_KEY"] = "sk-your-key"  # Replace with actual key

# 1. Read text file
with open("extracted_terms_and_conditions.txt", "r") as f:
    text_data = f.read()

# %%
# 2. Create semantic chunks
text_splitter = SemanticChunker(OpenAIEmbeddings())
document_chunks = text_splitter.create_documents([text_data])

# 3. Convert to pandas DataFrame
chunk_list = [chunk.page_content for chunk in document_chunks]
df = pd.DataFrame({"text_chunks": chunk_list})
# %%
# 4. Upload to Databricks
# add a column sec_id to the dataframe make it monotonically increasing
df["sec_id"] = range(1, len(df) + 1)

spark = (
    DatabricksSession.builder.profile("DOGFOOD").remote(serverless=True).getOrCreate()
)

spark_df = spark.createDataFrame(df)

# Write to Delta Lake table with sec_id as the primary key
spark_df.write.format("delta").option("primaryKey", "sec_id").saveAsTable(
    "main.sgfs.axle_fuel_card_terms_and_conditions"
)


spark.sql(
    "ALTER TABLE `main`.`sgfs`.`axle_fuel_card_terms_and_conditions` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
