import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
import nltk
import faiss
import pandas as pd
import numpy as np
# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

class semantic_search:
    def __init__(self,image):
        self.image = image
        self.index = None
        self.words = None
        self.bbxs = None
    def get_text_and_bounding_boxes(self):
        
        data = pytesseract.image_to_data(self.image, output_type=pytesseract.Output.DICT)
        words = []
        bounding_boxes = []

        for i in range(len(data['text'])):
            word = data['text'][i]
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
     
            if len(word) > 1:
                words.append(word)
                bounding_boxes.append((x, y, w, h))
        
        return words, bounding_boxes

    def get_embeddings(self,text):
        return np.mean(model.encode(list(set(re.findall('[^!?。.？！]+[!?。.？！]?', text))) ), axis=0)
   
    # def get_embeddings(self,text):

    #     # Tokenize and get BERT embeddings
    #     encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    #     with torch.no_grad():
    #         model_output = bert_model(**encoded_input)
        
    #     # Use last hidden states (embeddings)
    #     embeddings = model_output.last_hidden_state.squeeze()[0]
    #     return embeddings

    def preprocess(self,sentence):
        sentence=str(sentence)
        sentence = sentence.lower()
        sentence=sentence.replace('{html}',"") 
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', sentence)
        rem_url=re.sub(r'http\S+', '',cleantext)
        rem_num = re.sub('[0-9]+', '', rem_url)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem_num)  
        filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
        stem_words=[stemmer.stem(w) for w in filtered_words]
        lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
        return " ".join(filtered_words)
    
    def get_word_vectors(self):
        # words = []
        # sentences = self.get_text().split('\n')
        # for sent in sentences:
        #     # tokens = nltk.word_tokenize(sent)
        #     sent = (self.preprocess(sent))
        #     words.extend(sent.split(' '))
        words,bbxs = self.get_text_and_bounding_boxes()
        self.words = []
        self.bbxs = []
        word_embs = []
        for i,word in enumerate(words):
            word_emb = self.get_embeddings(word).reshape(1,-1)
            if word_emb.shape[1]>1:
                self.words.append(word)
                self.bbxs.append(bbxs[i])
                word_embs.append(word_emb)
        
        word_vectors = np.concatenate(word_embs,axis=0)
        return word_vectors
    
    def add_vectors_2_fiass(self,word_vectors):
        vector_dimension = word_vectors.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        self.index = faiss.IndexFlatIP(vector_dimension)
        faiss.normalize_L2(word_vectors)
        # word_vectors = word_vectors / np.linalg.norm(word_vectors, axis=1, keepdims=True)
        self.index.add(word_vectors.astype(np.float32))
        return self.index
    
    def search_word(self,search_word):
        df = {}
        search_vector = self.get_embeddings(search_word).reshape(1,-1)
        # _vector = (search_vector)/np.linalg.norm(search_vector, axis=1, keepdims=True)
        faiss.normalize_L2(search_vector)
        k = self.index.ntotal
        m = k
        distances, ann = self.index.search(search_vector, k=k)
        top_10_words = [self.words[i] for i in ann[0][:m]]
        df['query'] = search_word
        df['sim_words'] = top_10_words
        df['distance'] = distances[0,:m].tolist()
        df['indexes'] = ann[0,:m].tolist()
        df['bbx'] = [self.bbxs[i] for i in ann[0][:m]]
        return df

    def infer(self,query):
        word_vectors = self.get_word_vectors()
        self.index = self.add_vectors_2_fiass(word_vectors)
        res = self.search_word(query)
        df = pd.DataFrame.from_dict(res)
        # df = df[:5]
        # df = df[df['distance']>0.90][:]
        df = df[:10]
        res = df.to_dict(orient='list')
        return res
        
        
                
        