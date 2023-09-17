import streamlit as st
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

st.title("What's True: using Langchain SimpleSequentialChain")
st.markdown("ask anything you have in mind!!")

# API = YOUR_API_KEY 
if API:
    llm = OpenAI(temperature=0.8, openai_api_key=API)
else:
    st.warning("Enter your OpenAI API-Key. Get your key from [here](https://platform.openai.com/account/api-keys).\n")

user_question = st.text_input("Enter your question:",
                              placeholder="Cyanobacteria can perform photosynthesis, are they considered as plants?")

if st.button("seach", type="primary"):
    template = "{question}\n\n"
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """
    {statement}
    Make points that would be used to give solution for it.\n\n"""
    prompt_template = PromptTemplate(input_variables=['statement'], template=template)
    assumption_chain = LLMChain(llm=llm, prompt=prompt_template)
    assumption_chain_seq = SimpleSequentialChain(chains=[question_chain, assumption_chain], verbose=True)

    template = """Here is a bullet point list of assertions:
    {assertions}
    consider it and make solution out of it\n\n"""
    prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
    fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
    fact_checker_chain_seq = SimpleSequentialChain(chains=[question_chain, assumption_chain, fact_checker_chain], verbose=True)

    template = """In light of the above solution, give a perfect and simple solution'{}'""".format(user_question)
    template = """{facts}\n""" + template
    prompt_template = PromptTemplate(input_variables=["facts"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template)
    overall_chain = SimpleSequentialChain(chains=[question_chain, assumption_chain, fact_checker_chain, answer_chain], verbose=True)

    st.success(overall_chain.run(user_question))
