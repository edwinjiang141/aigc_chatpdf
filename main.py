
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from param_config import  ParseConfig,ArgumentParser
from filetype import guess
import openai
from llm_ora import Oracle_llm
from ai_translate import AiTranslate
from ai_knowledge import ai_knowledge
from langchain.chains.question_answering import load_qa_chain
from translation_chain import TranslationChain,PDFTranslator
import gradio as gr

from fastapi import FastAPI
from pydantic import BaseModel, Field

os.environ["TESSDATA_PREFIX"] = "C:\Program Files\Tesseract-OCR"
# print(os.environ["OPENAI_API_KEY"])

def launch_gradio():

    with gr.Blocks() as iface:
        gr.Markdown("AIGC 工具")
        
        with gr.Tab("知识图谱"):
            with gr.Column():
                file_inputs=gr.File(label="上传文件(PDF、Word、TXT)")
                know_class=gr.Dropdown(["SQL 优化","故障处理"],label="图谱类别")
                # allow_flagging="never"
                file_button=gr.Button("知识图谱转换及存储")

        with gr.Tab("AI 搜索"):
            with gr.Column():
                text_input=gr.Textbox(label="问题描述", value=" ")
                text_output=gr.Textbox(label="答案", value=" ")
                doc_class=gr.Dropdown(["SQL 优化","故障处理"],label="文档类别")
                ask_button=gr.Button("搜索...")
        
        with gr.Tab("AI 翻译"):
            with gr.Row():
                with gr.Column(scale=1):
                    source_lang=gr.Dropdown(["英文", "中文"],label="源语言")
                    dest_lang=gr.Dropdown(["英文", "中文"],label="目标语言")
                with gr.Column(scale=2):
                    translate_input=gr.Textbox(label="原文", placeholder="带翻译文本",value=" ")
                    translate_output=gr.Textbox(label="AI翻译", lines=10,value=" ")
            with gr.Row():
                    translate_button=gr.Button("翻译")
        
        with gr.Tab("AI SQL"):
             with gr.Column():
                query_input=gr.Textbox(label="问题描述", value=" ")
                sql_output=gr.Textbox(label="SQL", lines=10,value=" ")
                gene_button=gr.Button("AI 生成")
        
         
        ask_button.click(chat_with_pdf,[text_input,doc_class],outputs=text_output)     
        file_button.click(save_data_fasis,[file_inputs,know_class])
        gene_button.click(char_with_aisql,[query_input],outputs=sql_output)
        translate_button.click(translate_sentence,[translate_input,source_lang,dest_lang],outputs=translate_output)
        
    iface.launch(share=True, server_name="0.0.0.0")

def initialize_search_config():
    # 解析命令行
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()

    global config 
    config = ParseConfig()
    config.initialize(args)

def save_data_fasis(file_inputs,know_class):

    ai_konw = ai_knowledge(file_inputs,None,know_class)
    results = ai_konw.save_data_fasis()
    return results

def chat_with_pdf(query,doc_class):

    ai_konw = ai_knowledge(None,query,doc_class)
    results = ai_konw.chat_with_file()
    return results

def char_with_aisql(query):
    llm_sql = Oracle_llm()
    results = llm_sql.ai_generate_sql(query)
    return results

def translate_sentence(query,source_lang,dest_lang):
    llm_translate = AiTranslate()
    results = llm_translate.run(query,source_lang,dest_lang)
    return results

if __name__ == "__main__":
    launch_gradio()

app = FastAPI()
class Query(BaseModel):
    query: str

@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = char_with_aisql#agent({"input": query})
    actual_content = content['output']
    return actual_content