from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import warnings
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import spacy
import pyLDAvis
import pyLDAvis.gensim
from nltk.corpus import stopwords
import pickle

warnings.filterwarnings("ignore")

stop_words = stopwords.words('english')

df = pd.read_csv('abcnews-date-text.csv', nrows=10000)

lb = LabelEncoder()

x = lb.fit_transform(df['headline_text'])
x = x.reshape((-1, 1))

K = range(1, 15)
sum_of_sq_dist = []
for i in K:
    km = KMeans(n_clusters=i)
    km.fit(x)
    sum_of_sq_dist.append(km.inertia_)

fig = go.Figure(go.Scatter(x=list(K), y=sum_of_sq_dist))
fig.update_layout(
    title="Elbow plot",
    xaxis_title="Cluster value",
    yaxis_title="inertia",
    font=dict(
        size=18
    )
)
# fig.show()
# plot(fig)

km = KMeans(n_clusters=4)
km.fit(x)

y = km.transform(x)

cluster = km.predict(x)

data = pd.DataFrame({'headline_text': list(lb.inverse_transform(x)), 'cluster': cluster})

cluster0 = " ".join(data[(data['cluster'] == 0)]['headline_text'].tolist())
cluster1 = " ".join(data[(data['cluster'] == 1)]['headline_text'].tolist())
cluster2 = " ".join(data[(data['cluster'] == 2)]['headline_text'].tolist())
cluster3 = " ".join(data[(data['cluster'] == 3)]['headline_text'].tolist())

dataset = [cluster0, cluster1, cluster2, cluster3]


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


data_words = list(sent_to_words(dataset))

print(data_words[:1])

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words and len(word) > 4] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        if len(sent) > 4:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


data_words_nostops = remove_stopwords(data_words)

data_words_bigrams = make_bigrams(data_words_nostops)

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

id2word = corpora.Dictionary(data_lemmatized)

texts = data_lemmatized

corpus = [id2word.doc2bow(text) for text in texts]

print(corpus[:1])

corpora.Dictionary.save(id2word, 'id.sv')


with open('corpus.sv', 'wb') as f:
    pickle.dump(corpus, f)

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=4,
                                            random_state=42,
                                            update_every=1,
                                            chunksize=100,
                                            passes=20,
                                            alpha='auto',
                                            per_word_topics=True)

lda_model.save('lda.sv')

from gensim.models import CoherenceModel

print(lda_model.print_topics())
doc_lda = lda_model[corpus]

print('\nPerplexity: ', lda_model.log_perplexity(corpus))

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

# pyLDAvis.show(vis)
with open('lda_vis.html', 'w') as f:
    pyLDAvis.save_html(vis, f)
