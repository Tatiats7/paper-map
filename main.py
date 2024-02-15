from embeddings import check_and_add_embeds, find_similar_papers
from consts import template
from models import get_quantized_mistral_config
from langchain import PromptTemplate, LLMChain


root_path = "/content/drive/MyDrive/papers_for_workshop/"
data = check_and_add_embeds(root_path)
llm = get_quantized_mistral_config()

for i in range(10):
  print("Ask a question")
  question_p = input()
  if question_p == "exit":
    break
  context_p, source_papers = find_similar_papers(question_p, data)
  context_p = ""
  prompt = PromptTemplate(template=template, input_variables=["question","context"])
  llm_chain = LLMChain(prompt=prompt, llm=llm)
  response = llm_chain.run({"question":question_p,"context":context_p})
  print(response)
  print("source papers: ", source_papers)