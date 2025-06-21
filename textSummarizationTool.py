import nltk
nltk.download('punkt_tab')
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

text = """
In recent years, artificial intelligence has seen tremendous growth and impact across various industries. 
From healthcare to transportation, AI-powered solutions are being adopted to enhance efficiency, reduce 
human error, and uncover insights from vast amounts of data. In the healthcare sector, for example, machine 
learning algorithms can analyze medical images with high precision, sometimes even outperforming radiologists. 
Similarly, AI-driven predictive models assist in early diagnosis of diseases, potentially saving lives. 
Beyond healthcare, self-driving cars are becoming more viable due to advancements in computer vision and 
deep learning. In business, AI chatbots are revolutionizing customer support by providing instant responses 
and personalized recommendations. Despite its benefits, the rise of AI also raises ethical concerns, such as 
data privacy, algorithmic bias, and the displacement of human workers. Governments and institutions are 
working on frameworks to ensure that the development and deployment of AI remain fair, transparent, and 
beneficial to all. As we move forward, continuous research, regulation, and public awareness will be crucial 
to harness the full potential of artificial intelligence while mitigating its risks.
"""

parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = LsaSummarizer()
summary = summarizer(parser.document, 3)  # Number of sentences

for sentence in summary:
    print(sentence)
