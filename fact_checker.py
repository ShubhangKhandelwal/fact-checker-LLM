import streamlit as st
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

def main():
    st.title("What's True: using Langchain SimpleSequentialChain")
    st.markdown("Ask anything you have in mind!!")

    API = st.text_input("Enter your OpenAI API-Key:", type="password")
    
    if not API:
        st.warning("Enter your OpenAI API-Key. Get your key from [here](https://platform.openai.com/account/api-keys).\n")
        return

    llm = OpenAI(temperature=0.8, openai_api_key=API)

    user_question = st.text_input(
        "Enter your question:",
        placeholder="Cyanobacteria can perform photosynthesis, are they considered as plants?"
    )

    if st.button("Search", type="primary"):
        run_langchain(user_question, llm)

def run_langchain(user_question, llm):
    question_chain = create_llm_chain(llm, "question", "{question}\n\n")
    assumption_chain = create_llm_chain(llm, "statement", "{statement}\nMake points that would be used to give a solution for it.\n\n")
    fact_checker_chain = create_llm_chain(llm, "assertions", "Here is a bullet point list of assertions:\n{assertions}\nConsider it and make a solution out of it\n\n")
    answer_chain = create_llm_chain(llm, "facts", "{facts}\nIn light of the above solution, give a perfect and simple solution'{user_question}'")

    chains = [question_chain, assumption_chain, fact_checker_chain, answer_chain]
    overall_chain = SimpleSequentialChain(chains=chains, verbose=True)

    st.success(overall_chain.run(user_question))

def create_llm_chain(llm, input_variable, template):
    prompt_template = PromptTemplate(input_variables=[input_variable], template=template)
    return LLMChain(llm=llm, prompt=prompt_template)

if __name__ == "__main__":
    main()
