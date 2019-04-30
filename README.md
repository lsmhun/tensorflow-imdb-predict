# Tensorflow IMDb prediction 

Simple text classification based on Tensorflow IMDb data. Used simple technologies: Python 3, Tensorflow 2 alpha, Flask, Vue.js 

## Screenshot
![IMDb predict screenshot](/docs/imdb_movie_predict_01.png)

## See
Articles and howtos. I would recommend http://www.jtechlog.hu if you can read hungarian text.
 - https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
 - http://www.jtechlog.hu/2018/11/04/python-build.html
 - https://www.tensorflow.org/alpha/tutorials/keras/save_and_restore_models
 - https://stackoverflow.com/questions/45735070/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring
 - https://www.tensorflow.org/alpha/tutorials/text/text_classification_rnn

## Installation

You need Python3 installed. Then you need virtualenv.

```bash
$ make
$ source venv/bin/activate
$ ./run.sh
```

## Commands

You need Python3 installed. Then you need virtualenv. Some additional things
```bash
#### TRAIN is slow ####
$ make train
$ make test
$ make pylint
$ make sonar
```

