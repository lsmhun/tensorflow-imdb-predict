#!/usr/local/bin/python3
from imdb_ml_service import ImdbMlService
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('! WARNING! Train process is a long running task. On MacBookAir 2013, it ran 7 hours !')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
IMDB_ML_SERVICE = ImdbMlService()
IMDB_ML_SERVICE.train_save_model()
