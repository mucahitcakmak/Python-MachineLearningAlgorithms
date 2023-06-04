
# Kütüphaneler

# NLTK - Natural Language Tool Kit
# İstatistiksel ve sembolik DDİ - İngilizce

# Spacy 
# Ürün odaklı kullanılır. Sinir ağ modellerini de kapsar.

# Zemberek
# Türkçe DDİ | Python ve Java ile kullanılabilir

"""
"""

# Token nedir ? 
# İşlem görmemiş cümleler, kelimeler, sayılar ve somboller olabilir
# Cümleyi cümleden kelimelere ve noktalama işaretlerine göre ayırmaya tokenize etme işlemi denir

"""
"""

# Gövdeleme
# Kelimelerin köklerini bulmamız için gövdeleme yaparız (kök - stem)
# Kelimelerin kökünü bulmamızı sağlar | ekmekler = ekmek

# Baş Sözcük Çıkarma
# Kelimenin çekim ekimlerini atmak ve sözlükte olan haline ulaşmaktır | köküne değil
# civizilations = civizilation

"""
"""

# POSTagsing | Sözcük işaretleme
# POS - Part of speech - sözcük türü
# Sözcük türleri - Sıfat, nesne, yüklem ve bağlaç vb.
# Amaç : her kelimenin sözcük türünü işaretlemek(bulmak).
# Araştır : Universal Dependencies - Türkçe

# NER - Name Entity Recognition | Varlık İsmi Tanıma
# Metinde geçen varlık isimlerini işaretleme
# Varlık isimleri : Kişi, Organizasyon, Lokasyon, Tarih, Zaman ve Ülke oluşabilir
# Facebook = Organization | Germany = Location | Mustafa Kemal Atatürk = Person | bir varlık ismidir

"""
"""

# Metin Önişleme | Text Preprocessing ------------

# A - Metin Temizleme

# 1- Noktalama işaretlerinin ve özel karakterlerin kaldırılması 

# 2- Etkisiz kelimelerin kaldırılması | kendimizin tanımladığı veya hazır tanımlanmış
# etkisiz kelimeler vardır (stop words) | örn: ve, bir, ama, için | cümlede anlamı olmayan kelimeler denir bunlara

# 3- Sık ve nadir kullanılan kelimelerin çıkarılması

# 4- Kuraldışı kelimelerin düzeltilmesi
# way kelimesi gibi

# 5- Yazım Hataları

# 6- Gövdeleme ve Başsözcük çıkarmada metin önişlemede de kullanılması gereken özelliklerdir

"""
"""

# Metin Öznitelik Çıkarımı | Text Feature Extraction ------------

# A- Temel İstatistikler
# Kelimelerin değerlendirme sayılarına göre en çok kullanımına göre çıkarım yapar
# Mesela 5 yıldızlı bir cümlede güzel kelimesi iyi anlama gelirken 
# 1 yıldızlı cümledeki berbat kötü anlama gelir ve aralarında fark olduğunu anlar

# B- Bag Of Words (BOW) | Kelime Çantası Modeli 
# Kelime histogramları çıkarılır
# Kelimelerin kaç defa tekrar edildiği sayılır (tam anlamadım)

# C- Terim Frekansı * Ters Döküman Frekansı (TF*IDF Skoru)
# Tekrarlanmayan kelimelerin görülme sayısının metindeki toplam görülme sayısına oranıdır

# D- N-gram Modeli | sırasıyla hepsi
# 1gram, 2gram, 3gram
# Bir modeldeki ngram uzunluğu o modelin kelime bağlam uzunluğunu gösterir 
# 1 gram 2 gram kadar fazla bağlam bilgisi taşımaz.
# Unigram : P(teşekkür|çok) = n(çok teşekkür)/n(çok) teşekkür kelimesinin çok kelimesinden 
# sonra gelme olasılığını gösterir.
# gram arttıkça cümleyi daha çok alıyo

"""
"""

# Metin Sınıflandırma | Text Classification ------------

# 1- Metin Temsili
# Metinlerin sıfır ve birlere dönüştürülme olayı
# Merkez kelime vardır

# A- Frekans Tabanlı Temsiller
# Bow, TF-IDF ve Eş oluşum matrisleri (co-occuerence matrix)

# B- Tahminleme Tabanlı Temsiller
# Word2vec ve GloVe (Global Vectors for word representation)
# Kelimelerin vektörel yakınlıklarına göre hareket eder
# Benzerlik hesabı ile kelime bulma

"""
"""

# Konu Modelleme | Topic Modelling ------------

# Gözetimsiz makine öğrenme tekniğidir
# Soyut konular içerir
# Birbirine yakın cümleler kümelenir kendine göre ve hava durumu vs diye sınıflara ayrılır
# Hissiyat analizi, spam filtreleme ve chatbots kullanılır
# Konu modelleme yöntemleri :
# 1- Gizli Anlamsal Analiz - LSA
# 2- Negatif Olmayan Matrix Faktorizasyonu
# 3- Gizli Dirichlet  Ayırımı - LDA
# 4- Pachinko  Dağılım Modeli

# Gizli Dirichlet  Ayırımı - LDA
# Verilerin bazı kısımlarını açıklar
# Belgedeki kelimelerden konuları bulmak

"""
"""

# RNN - Özyinelemeli  
# A- Dizilim Modelleri
# Girdi, Saklı, Çıktı Katmanları


# LSTM
# RNN uzun bağlılık olayını çözmek için önerilen modeller
# kaybolmayı da çözmek için öneriliyo
