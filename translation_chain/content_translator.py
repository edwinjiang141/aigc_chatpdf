from typing import Optional
from translation_chain import TranslationChain
from utils import LOG

class PDFTranslator:
    def __init__(self, model_name: str):
        self.translate_chain = TranslationChain(model_name)

    def translate_pdf(self,
                    # input_file: str,
                    # output_file_format: str = 'markdown',
                    content:str,
                    source_language: str = "Chinese",
                    target_language: str = 'English'):
        
        translation, status = self.translate_chain.run(content, source_language, target_language)
        print(translation)
        return translation