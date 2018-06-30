Note: This is a truncated version of the [original readme](https://github.com/Conchylicultor/DeepQA/blob/master/README.md)

## Installation

The program requires the following dependencies (easy to install using pip: `pip3 install -r requirements.txt`):
 * python 3.5
 * tensorflow (tested with v1.0)
 * numpy
 * CUDA (for using GPU)
 * nltk (natural language toolkit for tokenized the sentences)
 * tqdm (for the nice progression bars)

You might also need to download additional data to make nltk work.

```
python3 -m nltk.downloader punkt
```

## Data

The `data` folder is truncated to reduce the size.
The original training data can be downloaded from the [original repo](https://github.com/Conchylicultor/DeepQA)

## Running

### Chatbot

To train the model, simply run `main.py`. 
