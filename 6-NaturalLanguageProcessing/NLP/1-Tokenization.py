import nltk


nltk.download("punkt") # Mecbur
nltk.download('averaged_perceptron_tagger') # PosTags
nltk.download('maxent_ne_chunker') # Ner 
nltk.download('words') # Ner


text = """Ceren, çok bilgili arkadaşımdır."""


# Cümle Tokanize işlemi | Noktalardan bölme
from nltk import sent_tokenize # Cümle Tokenize
print("Cümle Bölme İşlemi".center(50, "-"))
sentences = sent_tokenize(text)
i = 0
for sent in sentences:
    i += 1
    print(i, sent)
print("Soon".center(50, "-"))


print(" ")


# Kelime Tokenize işlemi | Metin içinden kelimeleri alıyor
from nltk import word_tokenize # Kelime Tokenize
print("Kelime Bölme İşlemi".center(50, "-"))
sentences = word_tokenize(text)
s = []
for sent in sentences:
    s.append(sent)
print(s)
print("Soon".center(50, "-"))



# Gövdeleme | Kelimedeki köke oluşma | fishing = fish -- Sadece İngilizce
from nltk.stem.porter import *
porter_stemmer = PorterStemmer()
word = "civilizations"
stemWord = porter_stemmer.stem(word)
print(stemWord)



# Gövdeleme Türkçe 
# Porter2 geliştirilmiş ve diğer diller için de stem(kök) bulabiliyor bu bir önceki sadece ingilizce
from snowballstemmer import TurkishStemmer
turkish_stem = TurkishStemmer()
stemWord = turkish_stem.stemWord("ekmekler") # çiçeklikler
print(stemWord)


# Baş sözcük çıkarma
import spacy
nlp = spacy.load("en_core_web_sm")
word = nlp("civizilations")
print(word.lemma_)



# POSTags = Cümlelerdeki ["zarf","yüklem"] vs gibi şeyleri bulur
partOfSpeech = nltk.pos_tag(sentences)
# print(partOfSpeech)



# NER (Named Entity Recognition) = Varlık ismini bulmaya kalkışır ["Ülke","Tarih","Facebook"]
entities = nltk.chunk.ne_chunk(partOfSpeech)
print(entities)
"""
# Spacy ile NER
import spacy 
from spacy import displacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")
sentence = nlp("Micheal Jordan is a professor at Berkeley")
print(sentence)
"""


# Temel istatistikler = kelime sayısı, büyük harf, etkisiz eleman sayısı,
# sayısal ifade sayısı, cümle karakter sayısı gibi şeyleri bulabilmemizi sağlar
# Hissiyat analizi = stop word ler bu problem için işe yarar
# Hissiyat analizi 2 = Kelimelerin büyük harf sayısı cümledeki hissiyatı anlamamıza yardımcı olur


# Bag Of Words = herk kelimenin kaç defa tekrar ettiğine bakar
# dezavantajı vardır kelimelerin yerlerini hatırlamaz vs sadece sayısını alıyo


# TF*İDF score = terim sayısı 


# N-gram = 1gram(uni) | 2 gram(bi) | 3gram(tri)
# cümlelerin otomatik tamamlanması, otomatik yazım düzeltme, cümledeki dil bilgisi kontrolü


# Frekans tabanlı temsiller
# kelime çantası modeli, sayma vektörleri ve tf*idf 
# eş oluşum matrisler (occuerence matrix)


