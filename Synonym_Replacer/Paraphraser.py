import re
import Text_Characteristics.Text_Characteristics as tc
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import Common.library.Common as com


def adapt_complexity_and_polarity(adaptation_dto, mean_measures, n_iterations, epsilon, debug):
    sentences = com.calc_sentence_similarity(adaptation_dto.adapted_text())
    text = adaptation_dto.adapted_text()
    rel_polar = abs((adaptation_dto.text_measures()['SENT_ANAL']['POLAR'] - mean_measures['SENT_ANAL'][adaptation_dto.target_pub_type()]['POLAR']) / mean_measures['SENT_ANAL'][adaptation_dto.target_pub_type()]['POLAR'])
    rel_read = abs((
        adaptation_dto.text_measures()['READ'] - mean_measures['READ'][adaptation_dto.target_pub_type()]) / \
               mean_measures['READ'][adaptation_dto.target_pub_type()])

    curr_diff = rel_polar + rel_read
    text_characteristics = tc(adaptation_dto.target_pub_type())

    for s in sentences:
        if n_iterations == 0 or abs(curr_diff) <= epsilon:
            break

        paraphrases = generate_sequences(s)
        best_paraphrase = None
        best_paraphrase_diff = None

        for p in paraphrases:
            replaced_text = text
            replaced_text = re.sub(s, p, replaced_text)

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
                best_paraphrase_diff = curr_diff_with_para

        if best_paraphrase is not None and best_paraphrase != s and curr_diff > best_paraphrase_diff:
            text = re.sub(s, best_paraphrase, text)

            if debug:
                print("Replacing '", s, "' for '", best_paraphrase, "'")
                print("Relative difference after replacement: ", best_paraphrase_diff)
            curr_diff = best_paraphrase_diff
        n_iterations = n_iterations - 1
    return adaptation_dto


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_sequences(sentence):
    set_seed(42)
    _data_absolute_path = 'C:/Luka/School/Bachelor/Bachelor\'s thesis/Text_Adaptation/Data/'

    model = T5ForConditionalGeneration.from_pretrained(_data_absolute_path + 'Model/t5_paraphrase')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)
    model = model.to(device)

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
