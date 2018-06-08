library(caret)
library(tm)
library(SnowballC)
library(arm)
# Training data.
data <- c('Ikan Makarel Kaleng Bercacing.',
          'temuan cacing mati di produk ikan makarel dalam kemasan.',
          'Ikan makarel masuk dalam keluarga Scombridae.',
          'BPOM Beberkan 27 Merek Ikan Makarel Kaleng Bercacing.',
          'merek produk mengandung cacing itu antara lain ABC, ABT, Ayam Brand, Botan, CIP, Dongwon, Dr. Fish.',
          '27 merek yang diumumkan 16 merupakan produk impor, dan 11 merupakan produk dalam negeri.',
          'merek ikan makarel atau sarden kalengan positif mengandung parasit cacing atau cacing jenis Anisakis Sp.',
          'kan Kaleng Mengandung Cacing Juga Ditemukan di Aceh.',
          'BBPOM Aceh melakukan inspeksi mendadak di Suzuya Mal, Banda Aceh.',
          'Semua ikan kaleng makarel ditarik dari pasaran.).',
          'Ada Ribuan merek Kaleng Sarden Bercacing.',
          'Cacing Pita Hidup Dalam Kaleng Sarden.',
          'Cacing Pita mampu hidup dalam suhu tinggi.',
          'sampai saat ini,BPOM belum menarik produk yang mengandung cacing.',
          'merek yang mengandung cacing merupakan produk lokal.',
          'BPOM tidak memantau pelaksanaan penarikan dan pemusnahan.',
          'produk yang berasal dari China tersebut adalah sarden makarel merek dagang botan,cip,dan dr fish.',
          'terdapat 20 merek makarel yang bercacing.',
          'BPOM belum menguji makarel yang mengandung cacing.',
          'belum adanya penarikan di pasaran terkait makarel bercacing.')
corpus <- VCorpus(VectorSource(data))

# Create a document term matrix.
tdm <- DocumentTermMatrix(corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

# Convert to a data.frame for training and assign a classification (factor) to each document.
train <- as.matrix(tdm)
train <- cbind(train, c(0, 1))
colnames(train)[ncol(train)] <- 'y'
train <- as.data.frame(train)
train$y <- as.factor(train$y)
data
train
# Train.
fit <- train(y ~ ., data = train, method = 'bayesglm')

# Check accuracy on training.
predict(fit, newdata = train)

# Test data.
data2 <- c('Untuk menyelamatkan konsumen dan menghindari hal yang lebih buruk, BBPOM pun menarik produk tersebut dari peredaran.',
           'Bakteri yang ada di dalam makanan kaleng jenis sarden tersebut sangat membahayakan kesehatan, karena bisa menembus saluran pencernaan sehingga memicu kematian.',
           'di Bandar Lampung, Lampung, razia produk ikan dalam kemasan kaleng juga digelar di tiga swalayan besar.',
           '27 Merek Sarden Mengandung Cacing Ditarik dari Peredaran.',
           'BPOM RI telah memerintahkan untuk menarik 27 produk dari pasaran hingga audit konferensif dilakukan.',
           'BBPOM Kota Pekanbaru telah mengeluarkan peringatan keras agar importir tiga merek sarden yang terbukti mengandung cacing menarik produknya dari pasaran.',
           'Ada Ribuan merek Kaleng Sarden Bercacing yang ada di Semarang.',
           'Cacing Pita Hidup Dalam Kaleng Sarden yang sudah dibekukan.',
           'Cacing Pita mampu hidup dalam suhu tinggi ketika dimasak.',
           'sampai saat ini,BPOM belum menarik produk yang mengandung cacing berbahaya itu.',
           'merek yang mengandung cacing merupakan produk lokal.',
           'BPOM tidak memantau pelaksanaan penarikan dan pemusnahan produk makarel.')
corpus <- VCorpus(VectorSource(data2))
tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))
test <- as.matrix(tdm)

# Check accuracy on test.
predict(fit, newdata = test)

