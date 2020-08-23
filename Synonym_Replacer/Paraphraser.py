import re
import Text_Characteristics.Text_Characteristics as tc
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import Common.library.Common as com


def adapt_complexity_and_polarity(model, tokenizer, device, adaptation_dto, mean_measures, n_iterations, epsilon, text_characteristics, debug):
    sentences,_ = com.calc_sentence_similarity(adaptation_dto.adapted_text())
    text = adaptation_dto.adapted_text()
    rel_polar = abs((adaptation_dto.text_measures()['SENT_ANAL']['POLAR'] - mean_measures['SENT_ANAL'][adaptation_dto.target_pub_type()]['POLAR']) / mean_measures['SENT_ANAL'][adaptation_dto.target_pub_type()]['POLAR'])
    rel_read = abs((
        adaptation_dto.text_measures()['READ'] - mean_measures['READ'][adaptation_dto.target_pub_type()]) / \
               mean_measures['READ'][adaptation_dto.target_pub_type()])

    curr_diff = rel_polar + rel_read
    for s in sentences:
        if n_iterations == 0 or abs(curr_diff) <= epsilon:
            break
        sentences_result = com.split_into_sentences(text)

        paraphrases = generate_sequences(model, tokenizer, device, s[1])
        best_paraphrase = None
        best_paraphrase_text = None
        best_paraphrase_diff = None

        for p in paraphrases:
            replaced_list = [p if x == s[1] else x for x in sentences_result]
            replaced_text = " ".join(replaced_list)

            curr_polar_with_para = text_characteristics.calc_polarity_scores(replaced_text)
            curr_read_with_para = com.flesch_reading_ease(replaced_text)

            rel_polar_with_para = abs((
                curr_polar_with_para - mean_measures['SENT_ANAL'][adaptation_dto.target_pub_type()]['POLAR']) / \
                                 mean_measures['SENT_ANAL'][adaptation_dto.target_pub_type()]['POLAR'])
            rel_read_with_para = abs((curr_read_with_para - mean_measures['READ'][adaptation_dto.target_pub_type()]) / \
                                mean_measures['READ'][adaptation_dto.target_pub_type()])
            curr_diff_with_para = rel_polar_with_para + rel_read_with_para

            if best_paraphrase is None or (curr_diff_with_para < best_paraphrase_diff):
                best_paraphrase = p
                best_paraphrase_text = replaced_text
                best_paraphrase_diff = curr_diff_with_para

        if best_paraphrase is not None and best_paraphrase != s[1] and curr_diff > best_paraphrase_diff:
            text = best_paraphrase_text

            if debug:
                print("Replacing '", s[1], "' for '", best_paraphrase, "'")
                print("Relative difference after replacement: ", best_paraphrase_diff)
            curr_diff = best_paraphrase_diff
        n_iterations = n_iterations - 1
    adaptation_dto.adapted_text(text)

    return adaptation_dto


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_sequences(model, tokenizer, device, sentence):
    set_seed(42)

    text = "paraphrase: " + sentence + " </s>"

    max_len = 256

    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=10
    )
    final_outputs = []
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    return final_outputs
