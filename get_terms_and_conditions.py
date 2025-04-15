#%%
import base64
import io
from dotenv import load_dotenv
import os
from databricks_langchain import ChatDatabricks
import fitz
from PIL import Image
# from IPython.display import Image as IPImage
# from IPython.display import display
from tqdm import tqdm
from langchain_core.messages import HumanMessage

load_dotenv()

def pdf_page_to_base64(pdf_path: str, page_number: int):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number - 1)  # input is one-indexed
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_total_pages(pdf_path: str) -> int:
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    pdf_document.close()
    return total_pages

def extract_text_from_all_pages(pdf_path: str, output_file: str):
    """
    Extract text from all pages of a PDF and save to a text file.
    
    Args:
        pdf_path: Path to the PDF file
        output_file: Path to save the extracted text
    """
    llm = ChatDatabricks(model="databricks-claude-3-7-sonnet",
                         temperature=0,
                         api_key=os.getenv("DATABRICKS_TOKEN"),
                         host=os.getenv("DATABRICKS_HOST"))
    
    total_pages = get_total_pages(pdf_path)
    print(f"Total pages in PDF: {total_pages}")
    
    query = """What is the text on the page? 
    Your job is to extract the text from the page and your accuracy should be as high as possible.
    Only return nicely formatted text, in a way that is easy to read. 
    Don't change the style or the content or entity names. 
    You might run into business names, don't change them. 
    For example, "One9 Fuel Network", "Pilot Flying J", "Axle/VOYAGER Fuel Card", etc.
    The page might be split by panes horizontally. 
    If so go from left to right and keep adding to the content. 
    Pay close attention to make sure you don't miss any text."""
    
    all_text = []
    
    for page_num in tqdm(range(1, total_pages + 1), desc="Processing pages"):
        try:
            # Convert page to base64
            base64_image = pdf_page_to_base64(pdf_path, page_num)
            
            # Optional: Display the current page
            # display(IPImage(data=base64.b64decode(base64_image)))
            
            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            )
            
            # Get response from LLM
            response = llm.invoke([message])
            
            # Add page number and content to results
            page_text = f"--- PAGE {page_num} ---\n{response.content}\n\n"
            all_text.append(page_text)
            
            print(f"Completed page {page_num}/{total_pages}")
            
        except Exception as e:
            print(f"Error processing page {page_num}: {str(e)}")
    
    # Write all extracted text to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("".join(all_text))
    
    print(f"Text extraction complete. Results saved to {output_file}")

# %%
# Execute the extraction
pdf_path = '2025-axle-fuel-card-terms-and-conditions.pdf'
output_file = 'extracted_terms_and_conditions.txt'

extract_text_from_all_pages(pdf_path, output_file)

# %%
# Display a sample of the extracted text (first 20 lines)
try:
    with open(output_file, 'r', encoding='utf-8') as f:
        sample_text = "".join([next(f) for _ in range(20)])
    print("Sample of extracted text:")
    print(sample_text)
    print(f"...\nFull text available in {output_file}")
except Exception as e:
    print(f"Error: {e}")
    print(f"File {output_file} not found or could not be read.")