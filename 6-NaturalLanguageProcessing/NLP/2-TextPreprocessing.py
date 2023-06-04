# Metin Temizleme

# 1
import string
message = "Dil işlemede kullanılan kütüphaneler: nltk, spacy, scikit-learn vb."
print(message.translate(str.maketrans("", "", string.punctuation)))

# 2
stopWords = ["acaba", "ve", "bir", "birçok", "ama", "için"] 
message = "Acaba metindeki dolgu kelimlerini bulmak ve temizlemek için ne yapılmalı"
s1 = set(stopWords)
s2 = set(message.lower().split())
print(s1.intersection(s2))