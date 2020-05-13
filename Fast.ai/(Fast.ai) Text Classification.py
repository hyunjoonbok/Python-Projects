# Language model using Wikipeida to predict the next word

from fastai.text import *   # Quick access to NLP functionality
from fastai.datasets import *

path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path/'texts.csv')

data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=32)

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.fit_one_cycle(1, 1e-2)

learn.unfreeze()
learn.fit(1, 1e-3)

learn.save_encoder('ft_enc')

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.fit(1, 1e-2)