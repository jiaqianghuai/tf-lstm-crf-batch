# tf-lstm-crf-batch

The tf-lstm-crf-batch tool is an implementation of a Named Entity Recognizer combined bi-lstm and crf based on tensorflow. Details about the model can be found at:https://arxiv.org/pdf/1603.01360.pdf


# Initial setup
To use the tool, you need Python 2.7, with Numpy and Tensorflow installed.


# Tag sentences

The fastest way to use the tool is to use one of the pretrained models:

```
 ./tagger.py --model models/your_model_name/ --saver models/saver/ --input input.txt --output output.txt
```

The input file should contain one sentence by line, and they have to be tokenized.


# Train a model

To train your own model, you need to use the train.py script and provide the location of the training, development and testing set:

```
 ./train.py --train train.txt --dev dev.txt --test test.txt
```

The training script will automatically give a name to the model and store it in ./models/ There are many parameters you can tune (CRF, dropout rate, embedding dimension, LSTM hidden layer size, batch_size, gpu, etc). To see all parameters, simply run:
```
 ./train.py --help
```

Input files for the training script: each word has to be on a separate line, and there must be an empty line after each sentence. A line must contain at least 2 columns, the first one being the word itself, the last one being the named entity. It does not matter if there are extra columns that contain tags or chunks in between.
