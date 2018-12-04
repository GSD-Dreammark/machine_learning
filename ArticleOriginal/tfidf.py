#!/usr/bin/env python
#coding=utf-8
import math
import re
import time
import codecs
from operator import itemgetter
class TfIdf:
  def __init__(self, corpus_filename=None, stopword_filename=None,
               DEFAULT_IDF=1.5):
      self.num_docs = 0
      self.term_num_docs = {}  # term : num_docs_containing_term
      self.stopwords = set([])
      self.idf_default = DEFAULT_IDF
      if corpus_filename:
          self.merge_corpus_document(corpus_filename)
      if stopword_filename:
          stopword_file = codecs.open(stopword_filename, "r", encoding='utf-8')
          self.stopwords = set([line.strip() for line in stopword_file])
  def merge_corpus_document(self, corpus_filename):
    """slurp in a corpus document, adding it to the existing corpus model
    """
    corpus_file = codecs.open(corpus_filename, "r", encoding='utf-8')
    # Load number of documents.
    line = corpus_file.readline()
    self.num_docs += int(line.strip())
    # Reads "term:frequency" from each subsequent line in the file.
    for line in corpus_file:
      tokens = line.rsplit(":",1)
      term = tokens[0].strip()
      try:
          frequency = int(tokens[1].strip())
      except Exception as err:
          if line in ("","\t"):
              #catch blank lines
              print("line is blank")
              continue
          else:
              raise
      if self.term_num_docs.has_key(term):
        self.term_num_docs[term] += frequency
      else:
        self.term_num_docs[term] = frequency
  def train_model(self, path):
    """Add terms in the specified document to the idf dictionary."""
    num=0
    self.term_num_docs={}
    for line in open(path):
      num+=1
      lines=line.split()
      lines=set(lines)
      for word in lines:
        if word in self.term_num_docs:
          self.term_num_docs[word] += 1
        else:
          self.term_num_docs[word] = 1
    self.num_docs=num
  def save_corpus_to_file(self, idf_filename):
    """Save the idf dictionary and stopword list to the specified file."""
    output_file = codecs.open(idf_filename, "w", encoding='utf-8')
    output_file.write(str(self.num_docs) + "\n")
    for term, num_docs in self.term_num_docs.items():
       output_file.write(term + ": " + str(num_docs) + "\n")

    # sorted_terms = sorted(self.term_num_docs.items(), key=itemgetter(1),
    #                       reverse=True)
    # stopword_file = open(stopword_filename, "w")
    # for term, num_docs in sorted_terms:
    #   if num_docs < STOPWORD_PERCENTAGE_THRESHOLD * self.num_docs:
    #     break
    #   stopword_file.write(term + "\n")
  def get_num_docs(self):
    """Return the total number of documents in the IDF corpus."""
    return self.num_docs
  def get_idf(self, term):
    if not term in self.term_num_docs:
      return self.idf_default
    return math.log(float(1 + self.get_num_docs()) /(1 + self.term_num_docs[term]))
  def word_tfidf(self, line,num):
      lis0=[]
      lines=line.split()
      if len(lines)>num :
         tfidf = {}
         for word in lines:
             mytf = float(line.count(word)) / len(lines)
             myidf = self.get_idf(word)
             tfidf[word] = mytf * myidf
         word_tf=sorted(tfidf.items(), key=itemgetter(1), reverse=True)
         for i,item in enumerate(word_tf):
            if i>(num-1):
                break
            lis0.append(item[0])
         keyword=' '.join(lis0)
         return  keyword
  def loda_model(self,corpus_filename):
      for i,line in enumerate(open(corpus_filename,encoding='utf-8')):
          if  i==0:
            self.num_docs=int(line.strip())
          else:
            lines=line.split(':')
            self.term_num_docs[lines[0]]=int(lines[1].strip())
  def one_tfidf(self,line,num,type=0):#line输入为分词结果
      if type==0 :
        lis0=[]
        lines=line.split()
        tfidf = {}
        for word in lines:
            mytf = float(line.count(word)) / len(lines)
            myidf = self.get_idf(word)
            tfidf[word] = mytf * myidf
        word_tf=sorted(tfidf.items(), key=itemgetter(1), reverse=True)
        for i,item in enumerate(word_tf):
            if i>(num-1):
                break
            lis0.append(item[0])
        st=' '.join(lis0)
        return st
      elif type==1:
        lines = line.split()
        if len(set(lines)) > 20:
            tfidf = {}
            for word in lines:
                mytf = float(line.count(word)) / len(lines)
                myidf = self.get_idf(word)
                tfidf[word] = mytf * myidf
            word_tf = sorted(tfidf.items(), key=itemgetter(1), reverse=True)
            return word_tf[0:20]
        else:
            return 1
# if __name__ == '__main__':
#     Tf = tfidf.TfIdf()  # tfidf
#     Tf.loda_model('corpus_19wan.txt')


