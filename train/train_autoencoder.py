import math
import os
import random
import re
import sys
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import unicodedata
from torch import optim, nn

from train import batch_helper

sys.path.insert(0, os.getcwd() + '/models/')  # to import modules in models
sys.path.insert(0, os.getcwd() + '/data/')  # to import modules in models

import models.autoencoder as autoencoder
from data.lang import Lang

# needed for pickle
from data.podcast_processor import PodcastEpisode
from data.arxiv_processor import ResearchArticle
from data.create_extractive_label import PodcastEpisodeXtra, ResearchArticleXtra



ITERATIONS = 15000
PRINT_ITERATIONS = 500
MAX_LENGTH = 1000
TEACHER_FORCING_RATIO = 0.5

SOS_TOKEN = 0
EOS_TOKEN = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(input_type: str, path: str) -> None:
    input_lang, output_lang, pairs = prepareData(input_type, path)
    print(random.choice(pairs))

    hidden_size = 256
    encoder1 = autoencoder.EncoderRNN(input_lang.n_words, hidden_size, DEVICE).to(DEVICE)
    attn_decoder1 = autoencoder.AttnDecoderRNN(hidden_size, output_lang.n_words, DEVICE, dropout_p=0.1).to(DEVICE)

    print("Using device ", DEVICE)
    print("Starting training...")
    history = train(encoder1, attn_decoder1, ITERATIONS, input_lang=input_lang, output_lang=output_lang,
                    pairs=pairs, print_every=PRINT_ITERATIONS)
    print("Training ended.")

    print("Sample translations: \n" + translate_random(encoder1, attn_decoder1, pairs, input_lang, output_lang))

    print("Generating loss plot...")
    showPlot(history)


def train(encoder: autoencoder.EncoderRNN, decoder: autoencoder.AttnDecoderRNN, n_iters: int,
          input_lang: Lang, output_lang: Lang,
          pairs, print_every=1000, plot_every=100, learning_rate=0.01) -> List[float]:
    """
    Train the autoencoder.
    @param encoder: the encoder
    @param decoder: the decoder
    @param n_iters: the number of iterations
    @param input_lang: the input language
    @param output_lang: the output language
    @param pairs: the translation pairs
    @param print_every: number of iterations before an update print
    @param plot_every: number of points in the returned history list
    @param learning_rate: the learning rate of the optimizer
    @return: a list containing the average loss for every plot_every training iterations
    """
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(input_lang=input_lang, output_lang=output_lang, pair=random.choice(pairs))
                      for _ in range(n_iters)]
    criterion = nn.NLLLoss()

    for i in range(1, n_iters + 1):
        training_pair = training_pairs[i - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train_iteration(input_tensor, target_tensor, encoder,
                               decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters),
                                         i, i / n_iters * 100, print_loss_avg))

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return plot_losses


def train_iteration(input_tensor: torch.Tensor, target_tensor: torch.Tensor, encoder: autoencoder.EncoderRNN,
                    decoder: torch.nn.Module, encoder_optimizer: torch.optim,
                    decoder_optimizer: torch.optim, criterion, max_length: int = MAX_LENGTH) -> List[float]:
    """
    A single iteration of the training loop.
    @param input_tensor: the tensor holding the embedded input string
    @param target_tensor: the tensor holding the embedded desired output string
    @param encoder: the encoder
    @param decoder: the decoder
    @param encoder_optimizer: the encoder's optimizer algorithm
    @param decoder_optimizer: the decoder's optimizer algorithm
    @param criterion: the loss criterion used in backpropagation
    @param max_length: the maximum length of the sentence
    @return: the total loss of the training iteration
    """
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_TOKEN]], device=DEVICE)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_TOKEN:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def evaluate(encoder: autoencoder.EncoderRNN, decoder: torch.nn.Module, sentence: str, input_lang: Lang,
             output_lang: Lang, max_length=MAX_LENGTH) -> Tuple[List[str], torch.Tensor]:
    """
    Get the translation for an arbitrary sentence.
    @param encoder: the encoder
    @param decoder: the decoder
    @param sentence: the sentence to be translated
    @param input_lang: the input language
    @param output_lang: the output language
    @param max_length: the max length of the sentence
    @return: the translated words and attention mappings
    """
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_TOKEN]], device=DEVICE)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def translate_random(encoder: autoencoder.EncoderRNN, decoder: torch.nn.Module, pairs: List[List[str]],
                     input_lang: Lang, output_lang: Lang, n=10) -> str:
    """
    Translate N random sentences.
    @param encoder: the encoder
    @param decoder: the decoder
    @param pairs: the pairs from which the selection will occur
    @param input_lang: the input language
    @param output_lang: the output language
    @param n: the number of translations to be picked
    @return: A string representing the original, original translation, and generated translation of N random sentences
    """
    output = ""
    for i in range(n):
        pair = random.choice(pairs)
        output += ">" + pair[0]
        output += "=" + pair[1]
        output_words, attentions = evaluate(encoder, decoder, pair[0],
                                            input_lang=input_lang, output_lang=output_lang)
        output_sentence = ' '.join(output_words)
        output += "<" + output_sentence + "\n"

    return output


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters

def preprocess(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(path: str) -> List[List[str]]:
    """
    Read, preprocess and return a list containing the input documents.
    @param path: the path to the .bin file containing the input documents
    @return: a list containing the documents in pairs with themselves
    """
    print("Reading documents...")
    articles = batch_helper.load_articles(path)

    # train on the whole text
    pairs = [[" ".join(article.article_text), " ".join(article.article_text)] for article in articles]

    return pairs


def prepareData(input_type: str, path: str) -> Tuple[Lang, Lang, List[List[str]]]:
    """
    Read the input documents and use them to create the necessary language objects.
    @param input_type: "arxiv" or "pubmed" depending on the dataset used
    @param path: the path to the data .bin file
    @return: the input and output language objects and the input and output document pairs
    """

    # TODO: utilize validation data
    if input_type == "arxiv":
        train_data_path = "{}/arxiv_train.pk.bin".format(path)
        val_data_path = "{}/arxiv_val.pk.bin".format(path)
    elif input_type == "pubmed":
        train_data_path = "{}/pubmed_train.pk.bin".format(path)
        val_data_path = "{}/pubmed_val.pk.bin".format(path)
    else:
        raise NotImplementedError("Input type must be either 'arxiv' or 'pubmed', not " + input_type)

    # we will use a single language object as both input and output
    lang = Lang("article")

    pairs = readLangs(train_data_path)
    print("Read %s sentence pairs" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        lang.addSentence(pair[0])

    # debug
    print("Counted words:")
    print(lang.name, lang.n_words)

    return lang, lang, pairs


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python train_autoencoder.py input_type data_path")
        print("\tWhere input_type one of 'arxiv', 'pubmed' depending on the dataset used")
