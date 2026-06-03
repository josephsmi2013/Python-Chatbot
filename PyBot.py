
from dotenv import load_dotenv

import pandas as pd


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



#Load env file. Specifically the API key
load_dotenv()


#Python function to read the sample FAQ from a file using utf-8 encoding
#Returning the documents object should handle chunking which leads to better embeddings, better retrieval, better answers
def loadFAQ(filePath):
    df = pd.read_csv(filePath, encoding="cp1252").dropna()

    document = [
        f"Question: {str(row['Questions']).strip()}\n"
        f"Answer: {str(row['Answers']).strip()}"
        for _, row in df.iterrows()
    ]

    return document


#Python function to create the embeddings and store them using FAISS
def createDB(chunks):

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)

    return db

    
#Python function that creates a prompt from a template and passes it to the selected LLM (OpenAI's gpt-4.1 nano)
#It also generates the context from our data (stored as a vector DB) and builds the actual RAG chain
def createChain(db):

    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """ You are a Python FAQ chatbot. Your goal is to answer questions about Python. You may go outside your context to answer, but only if the question is directly related to python

        Context:
        {context}

        Question:
        {question}
        """
    )

    model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

    #Create the chain using our prompt, LLM, and retriever
    chain = ({"context": retriever, "question": lambda x: x} | prompt | model | StrOutputParser())

    return chain



########################################################
#Main app entry point
#Use double underscores to support Python name wrangling
#########################################################
if __name__ == "__main__":

    print("FAQ Chatbot Loading\n")

    #Load the sample FAQ data
    document = loadFAQ("Python FAQ Dataset.csv")

    #Store the document returned from the CSV
    db = createDB(document)

    #Create the FAQ chain
    chain = createChain(db)

    print("FAQ Chatbot Ready! Type 'exit' to quit at any time\n")

    #Loop allows us to accept user input until they quit
    while True:

        query = input("You: ")

        if query.lower() == "exit":
            break

        response = chain.invoke(query)
        print("Bot:", response, "\n")