from Tokenize import Interval, Document
import re

class Parser(object):
    """Classe parente pour tous les parsers"""
    def create(self):
        return self

    def read_file(self, filename: str) -> Document:
        with open(filename, 'r', encoding='utf-8') as fp:
            content = fp.read()
        return self.read(content)



class AmazonReviewParser(Parser):
    def read(self, content: str) -> Document:
        """Reads the content of an amazon data file and returns one document instance per document it finds."""
        
        documents = []
        docs = content.split("\n")
        list_overall = []
        list_review_text = []
        cpt = 0
        for line in docs:
            if cpt <= 10260:
                line = eval(line)
                review_text = line["reviewText"]
                overall = line["overall"]

                documents.append(Document.create_from_text(review_text))
                cpt+=1
                print(cpt)
      
        # Split lines and loop over them
        # Read json with: data = json.loads(line)
        # Instantiate Document object from "reviewText" and label from "overall"
     
        return documents