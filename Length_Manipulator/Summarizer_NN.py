import os
import statistics
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, WarmUp
import torch
import requests
import xml.etree.ElementTree as ET
from rake_nltk import Metric, Rake
from nltk.corpus import stopwords
from Common.classes.TextRank4Keyword import TextRank4Keyword
import Common.library.Common as com
from nltk import tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
import math
from Common.classes.RNN_Model import Prediction, Preprocessing


def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


def generate_some_text(model, tokenizer, device, input_str, text_len=250):
    cur_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0).long().to(device)

    model.eval()
    with torch.no_grad():
        for i in range(text_len):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0, -1],
                                           dim=0)  # Take the first(only one) batch and the last predicted embedding
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(),
                                            n=10)  # Randomly(from the given probability distribution) choose the next word from the top n words
            cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id],
                                dim=1)  # Add the last word

        output_list = list(cur_ids.squeeze().to('cpu').numpy())
        output_text = tokenizer.decode(output_list)
        return output_text
    return input_str


# https://huggingface.co/transformers/model_doc/bart.html?highlight=no_repeat_ngram_size

def summarize(adaptation_dto, mean_measures, epsilon, debug=False):
    init_text = adaptation_dto.adapted_text()
    tokens = tokenize.word_tokenize(init_text)
    n_blocks = math.floor(len(tokens) / 512)
    blocks_tokens = []
    for i in range(n_blocks):
        blocks_tokens.append(tokens[i * 512:(i + 1) * 512])

    blocks_tokens.append(tokens[(n_blocks) * 512:])
    blocks = []
    for b in blocks_tokens:
        tmp = ""
        for t in b:
            tmp = tmp + t + " "
        blocks.append(tmp)
    summary = ""
    if debug:
        print(len(blocks))
    a_tokens = round(math.floor(-epsilon * mean_measures["LEN"][adaptation_dto.target_pub_type()] \
                          + mean_measures["LEN"][adaptation_dto.target_pub_type()]) / n_blocks + 1)
    b_tokens = round(math.ceil(epsilon * mean_measures["LEN"][adaptation_dto.target_pub_type()] \
                         + mean_measures["LEN"][adaptation_dto.target_pub_type()]) / n_blocks + 1)
    count = 0
    for text in blocks:
        if debug:
            print(count)
        count = count + 1
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        preprocess_text = text.strip().replace("\n", "")
        t5_prepared_text = "summarize: " + preprocess_text

        tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors="pt").to(device)

        summary_ids = model.generate(tokenized_text,
                                     num_beams=4,
                                     no_repeat_ngram_size=2,
                                     min_length=b_tokens if a_tokens > b_tokens else a_tokens,
                                     max_length=a_tokens if a_tokens > b_tokens else b_tokens,
                                     early_stopping=False,
                                     repetition_penalty=2.5,
                                     length_penalty=1.50)

        summary = summary + tokenizer.decode(summary_ids[0], skip_special_tokens=False)

        if debug:
            print("Summarization done")
        # print("\n\n    ------------------------------- ORIGINAL TEXT -------------------------------    \n\n",
        #       adaptation_dto.adapted_text())

    # if debug:
        # print("\n\n    ------------------------------- SUMMARY -------------------------------    \n\n", summary)

    adaptation_dto.adapted_text(summary)

    if debug:
        print("Text measures after summarization: ", adaptation_dto.text_measures())

    return summary


def generate(adaptation_dto, mean_measures, debug):
    MODEL_EPOCH = 4
    text = adaptation_dto.adapted_text()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model = model.to(device)

    models_folder = "/content/drive/My Drive/Model/"

    model_path = os.path.join(models_folder, f"gpt2_{adaptation_dto.target_pub_type().lower()}_{MODEL_EPOCH}.pt")
    model.load_state_dict(torch.load(model_path))

    model.eval()
    if debug:
        print("Text measures before generation: ", adaptation_dto.text_measures())

    text = generate_some_text(model, tokenizer, device, text, round(abs(
        mean_measures['LEN'][adaptation_dto.target_pub_type()] - adaptation_dto.text_measures()['LEN'])))

    return text


def adapt_length(adaptation_dto, mean_measures, n_iterations, epsilon, debug=False):
    text_measures = adaptation_dto.text_measures()

    curr_diff = (text_measures['LEN'] - mean_measures['LEN'][adaptation_dto.target_pub_type()]) / \
                mean_measures['LEN'][adaptation_dto.target_pub_type()]

    if debug:
        print("Starting difference sum: ", curr_diff)

    while n_iterations > 0 and abs(curr_diff) > epsilon:
        if curr_diff > 0:
            text = summarize(adaptation_dto, mean_measures, epsilon, debug)
        else:
            text = generate(adaptation_dto, mean_measures, debug)

        length = len(tokenize.word_tokenize(text))
        curr_diff = (length - mean_measures['LEN'][adaptation_dto.target_pub_type()]) / mean_measures['LEN'][
            adaptation_dto.target_pub_type()]
        if debug:
            print("Relative difference after length manipulation: ", curr_diff)
        adaptation_dto.adapted_text(text)
        n_iterations = n_iterations - 1

    return adaptation_dto
