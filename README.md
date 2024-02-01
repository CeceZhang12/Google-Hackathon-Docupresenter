# Docupresenter
## Overview
DocuPresenter is a tool that allows you to create presentations using text extracted from PDF documents. This README provides a step-by-step guide on how to set up and use DocuPresenter.


## Setup
1. First, install the required Python packages by running the following commands
```
!pip install -U -q google-generativeai
!pip install --upgrade google-api-python-client
!pip install PyPDF2
!apt-get update
!apt-get install -y wkhtmltopdf
```
2. Import the necessary libraries and set up your API Key:
```
import google.generativeai as palm
import textwrap
import numpy as np
import pandas as pd
	
# Set your API Key
palm.configure(api_key='YOUR_API_KEY_HERE')
```
3. Choose a model for text embedding. The following code lists available models and selects one for text embedding.
```
models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
model = models[0]  # Choose the desired model  
```
## Retrieve PDF Files
1. Mount Google Drive to Google Colab to access your PDF files. Run the following code:
```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```
2. Extract text from PDF files in a specific folder and create a Pandas DataFrame to store it. Ensure that your PDF files are stored in the folder you specify (folder_path). The code reads and compiles its text content, removing newline characters for better formatting. 

_"shortened_text = cleaned_text.encode('utf-8')[:9900].decode('utf-8', errors='ignore')"_ 
At this moment, a characters limit of 9900 bytes is set since there is a maximum limit of 10000 bytes when embedding text. This processing step helps avoiding any potential errors. However, under ideal conditions, this API will be able to process a larger amount of text.
```
import os
import PyPDF2
import pandas as pd


folder_path = "/content/gdrive/MyDrive/Test"
pdf_dict = {}


# Check if the folder exists
if os.path.exists(folder_path):
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):  # Check if it's a PDF file
            pdf_path = os.path.join(folder_path, filename)


            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)


                # Extract text from each page
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()


                # Remove newline characters from the text
                cleaned_text = text.replace("\n", "")


                # Encode to bytes, extract the first 9000 bytes, and then decode
                shortened_text = cleaned_text.encode('utf-8')[:9900].decode('utf-8', errors='ignore')


                # Add the shortened text to the dictionary
                pdf_dict[filename] = shortened_text


else:
    print(f"Folder {folder_path} does not exist!")
# Convert the dictionary to a dataframe
df = pd.DataFrame(list(pdf_dict.items()), columns=['Filename', 'Text'])
df
```
3. Generate embeddings for the text and add them to the DataFrame.
```
# Get the embeddings of each text and add to an embeddings column in the dataframe
def embed_fn(text):
  return palm.generate_embeddings(model=model, text=text)['embedding']

df['Embeddings'] = df['Text'].apply(embed_fn)
df
```
## Query the Documents
1. Specify the topic for your query.
```
topic = "Physical Geography of Africa"
```
2. Create a function to find the most relevant passage related to the topic in your documents.
```
def find_best_passage(topic, dataframe):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = palm.generate_embeddings(model=model, text=topic)
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding['embedding'])
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['Text'] # Return text from index with max value
```
3. Query the documents to find the best passage.
```
passage = find_best_passage(topic, df)
passage
```
4. Create a prompt using the found passage and topic.
```
def make_prompt(topic, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""You are a helpful and informative bot that creates presentations using text from the reference passage included below. \
  I am a teacher for a group of 13-year-old students, please output markdown scripts
  If the passage is irrelevant to the presentation, you may ignore it.
  Topic: '{topic}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(topic=topic, relevant_passage=escaped)

  return prompt
```
## Generate a Presentation
1. Choose a text generation model and set parameters like temperature.
```
text_models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]

text_model = text_models[0]

temperature = 0.5
```
2. Generate a presentation using the prompt and the selected text generation model.
```
answer = palm.generate_text(prompt=prompt,
                            model=text_model,
                            temperature=temperature,
                            max_output_tokens=1000)
print(answer.result)

llm_output= answer.result
```
3. Convert the generated Markdown content to a PDF and save it to Google Drive.
```
from weasyprint import HTML
import markdown

# Ensure llm_output is a string and strip unnecessary characters if present.
llm_output = llm_output.strip("```").strip()

# Convert Markdown content to HTML
html_content = markdown.markdown(llm_output)

# HTML and CSS for the presentation-like format

# HTML and CSS for the presentation-like format
presentation_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Presentation</title>
    <style>
        @page {{
            size: A4 landscape;
            margin: 0mm;
        }}
        body {{
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: block;
        }}
        section {{
            width: 80%;
            max-width: 1280px;
            margin: 1cm auto;
            page-break-after: always;
            page-break-inside: avoid;
            display: block;
        }}
        h1, h2, h3, h4 {{
            text-align: center;
            margin-top: 0.5cm;
            margin-bottom: 0.5cm;
        }}
        p, li {{
            font-size: 24px;
            line-height: 1.5;
            text-align: left;
            margin-left: 10%;
            margin-right: 10%;
        }}
        ul, ol {{
            padding-left: 20px;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""

# Set the output file name
output_file_name = "presentation.pdf"

# Set the path to save the PDF file (modify as needed)
pdf_file_path = f"/content/gdrive/MyDrive/Test/{output_file_name}"

# Generate the PDF from the HTML string and save it to the specified path
HTML(string=presentation_html).write_pdf(pdf_file_path)

print(f"The presentation PDF has been created and saved to {pdf_file_path}.")
```
