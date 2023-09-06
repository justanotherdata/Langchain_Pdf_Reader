import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI



#Loading the openaikey
from dotenv import load_dotenv



def main():
    load_dotenv()
    print(os.getenv('OPENAI_API_KEY'))
    st.title('Ask me relevant questions from your pdf!')
    #Uploading pdf file here
    pdf = st.file_uploader("Upload Your Pdf Here", type = 'pdf')

    #Reading the file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()

        

        #Splitting the text
        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        #Encoding the chunks

        embeddings = OpenAIEmbeddings()
        doc_to_search_from = FAISS.from_texts(chunks, embeddings)

        #Taking quesry from user
        query = st.text_input("Enter Your question here")
        if query:
            rel_docs = doc_to_search_from.similarity_search(query)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            result = chain.run(input_documents=rel_docs, question=query)

            st.write(result)
            



    
if __name__ == '__main__':
    main()