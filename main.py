import json
import pandas as pd
import spacy
from PyDictionary import PyDictionary

import Common.classes.Text_Adaptation as ta
import Text_Characteristics.Text_Characteristics as text_char
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# text_characteristics = Text_Characteristics.Text_Characteristics(None, None)
# mean_measures = text_characteristics.mean_measures(True)
# variance_measures = text_characteristics.variance_map()
MODEL_EPOCH = 4

model_sum = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer_sum = T5Tokenizer.from_pretrained('t5-base')
device_sum = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_sum.to(device_sum)

models_folder = "/content/drive/My Drive/Model/"

model_para = T5ForConditionalGeneration.from_pretrained(models_folder + 't5_paraphrase')
tokenizer_para = T5Tokenizer.from_pretrained('t5-base')

device_para = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_para = model_para.to(device_para)

_data_absolute_path = '/content/Text_Adaptation_To_Context/Data/'

mean_measures = {
    "LEN": {'SOC_MED': 24.09, 'NEWS': 939.49, 'OFC_STMT': 923.15, 'RES_ARTCL': 4390.29},
    "SENT_ANAL":
        {
            "SOC_MED": {'POLAR': 0.04},
            "NEWS": {'POLAR': 0.05},
            "OFC_STMT": {'POLAR': 0.18},
            "RES_ARTCL": {'POLAR': 0.04}
        },
    "READ": {'SOC_MED': 61.00, 'NEWS': 47.72, 'OFC_STMT': 26.89, 'RES_ARTCL': 27.06}
}

variance_measures = {
    "LEN": {'SOC_MED': 4.35, 'NEWS': 694.94, 'OFC_STMT': 1480.81, 'RES_ARTCL': 5052.45},
    "SENT_ANAL":
        {
            "SOC_MED": 0.30,
            "NEWS": 0.12,
            "OFC_STMT": 0.22,
            "RES_ARTCL": 0. + 12
        },
    "READ": {'SOC_MED': 19.61, 'NEWS': 9.16, 'OFC_STMT': 22.66, 'RES_ARTCL': 10.91}
}

modes = ["PARA"]
for mode in modes:
    targets = ["RES_ARTCL"]
    for target in targets:
        # if (target == "SOC_MED" or target == "NEWS") and mode != "PARA":
        #     continue
        device_gen = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer_gen = GPT2Tokenizer.from_pretrained('gpt2-medium')
        model_gen = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        model_gen = model_gen.to(device_gen)

        model_path = os.path.join(models_folder, f"gpt2_{target.lower()}_{MODEL_EPOCH}.pt")
        model_gen.load_state_dict(torch.load(model_path))
        model_gen.eval()

        dictionary = PyDictionary()
        nlp = spacy.load("en_core_web_sm")

        sample_size = 50
        debug = False
        iter = 0

        tc = text_char(target)


        # def plus(dic1, dic2):
        #     return {
        #         "LEN": dic1["LEN"] + dic2["LEN"],
        #         "SENT_ANAL": {
        #             "POLAR": dic1["SENT_ANAL"]["POLAR"] + dic2["SENT_ANAL"]["POLAR"]
        #         },
        #         "READ": dic1["READ"] + dic2["READ"]
        #     }


        def adapt(dataset, orig_type):
            # sum_measures_init = {
            #         "LEN": 0,
            #         "SENT_ANAL": {
            #             "POLAR": 0
            #         },
            #         "READ": 0
            #     }
            # sum_measures_length = {
            #         "LEN": 0,
            #         "SENT_ANAL": {
            #             "POLAR": 0
            #         },
            #         "READ": 0
            #     }
            # sum_measures_result = {
            #         "LEN": 0,
            #         "SENT_ANAL": {
            #             "POLAR": 0
            #         },
            #         "READ": 0
            #     }
            print("Dataset length: " + str(len(dataset)))
            for count in range(len(dataset)):
                try:
                    count_file = count + iter*sample_size
                    text = dataset[count]
                    print(count_file)
                    output = open("/content/drive/My Drive/Data/Results_02/input_" + mode + "_" + target + "_" + orig_type + str(count_file) + ".txt", "w+", encoding='utf-8')
                    output.write(str(text))
                    output.close()

                    # init_measures = tc.calc_text_measures(text)
                    # sum_measures_init = plus(sum_measures_init, init_measures)
                    adapted_text, adapted_length = ta.adapt_text(mode, model_para, model_gen, model_sum, tokenizer_para, tokenizer_gen, tokenizer_sum, device_para, device_gen, device_sum, text, mean_measures, target, nlp, dictionary, tc, debug)
                    # adapted_length_measures = tc.calc_text_measures(adapted_length)
                    # sum_measures_length = plus(sum_measures_length, adapted_length_measures)

                    # adapted_measures = tc.calc_text_measures(adapted_text)
                    # sum_measures_result = plus(sum_measures_result, adapted_measures)

                    output = open("/content/drive/My Drive/Data/Results_02/out_" + mode + "_" + target + "_" + orig_type + str(count_file) + ".txt", "w+", encoding='utf-8')
                    output.write(str(adapted_text))
                    output.close()
                except Exception as e:
                    print("OMFG AN ERROOOOOR!!!! I don't give a shit, let's continue");
                    print(e)
                    continue

            # mean_measures_init = {
            #         "LEN": sum_measures_init["LEN"] / sample_size,
            #         "SENT_ANAL": {
            #             "POLAR": sum_measures_init["SENT_ANAL"]["POLAR"] / sample_size
            #         },
            #         "READ": sum_measures_init["READ"] / sample_size
            #     }
            #
            # mean_measures_init_rel = {
            #     "LEN": abs((mean_measures_init['LEN'] - mean_measures['LEN'][target]) / mean_measures['LEN'][target]),
            #     "SENT_ANAL": {
            #         "POLAR": abs((mean_measures_init['SENT_ANAL']['POLAR'] - mean_measures['SENT_ANAL'][target]['POLAR']) / mean_measures['SENT_ANAL'][target]['POLAR'])
            #     },
            #     "READ": abs((mean_measures_init['READ'] - mean_measures['READ'][target]) / mean_measures['READ'][target])
            # }
            #
            # mean_measures_result = {
            #         "LEN": sum_measures_result["LEN"] / sample_size,
            #         "SENT_ANAL": {
            #             "POLAR": sum_measures_result["SENT_ANAL"]["POLAR"] / sample_size
            #         },
            #         "READ": sum_measures_result["READ"] / sample_size
            #     }
            #
            # mean_measures_result_rel = {
            #     "LEN": abs((mean_measures_result['LEN'] - mean_measures['LEN'][target]) / mean_measures['LEN'][target]),
            #     "SENT_ANAL": {
            #         "POLAR": abs((mean_measures_result['SENT_ANAL']['POLAR'] - mean_measures['SENT_ANAL'][target]['POLAR']) / mean_measures['SENT_ANAL'][target]['POLAR'])
            #     },
            #     "READ": abs((mean_measures_result['READ'] - mean_measures['READ'][target]) / mean_measures['READ'][target])
            # }
            #
            # mean_measures_length = {
            #         "LEN": sum_measures_length["LEN"] / sample_size,
            #         "SENT_ANAL": {
            #             "POLAR": sum_measures_length["SENT_ANAL"]["POLAR"] / sample_size
            #         },
            #         "READ": sum_measures_length["READ"] / sample_size
            #     }
            #
            # mean_measures_length_rel = {
            #     "LEN": abs((mean_measures_length['LEN'] - mean_measures['LEN'][target]) / mean_measures['LEN'][target]),
            #     "SENT_ANAL": {
            #         "POLAR": abs((mean_measures_length['SENT_ANAL']['POLAR'] - mean_measures['SENT_ANAL'][target]['POLAR']) / mean_measures['SENT_ANAL'][target]['POLAR'])
            #     },
            #     "READ": abs((mean_measures_length['READ'] - mean_measures['READ'][target]) / mean_measures['READ'][target])
            # }

            return None, None, None, None, None, None # mean_measures_init, mean_measures_result, mean_measures_init_rel, mean_measures_result_rel, mean_measures_length, mean_measures_length_rel

        dict_init = {
            'OFC_STMT': None,
            'RES_ARTCL': None,
            'SOC_MED': None,
            'NEWS': None
        }

        dict_length = {
            'OFC_STMT': None,
            'RES_ARTCL': None,
            'SOC_MED': None,
            'NEWS': None
        }

        dict_result = {
            'OFC_STMT': None,
            'RES_ARTCL': None,
            'SOC_MED': None,
            'NEWS': None
        }

        dict_init_rel = {
            'OFC_STMT': None,
            'RES_ARTCL': None,
            'SOC_MED': None,
            'NEWS': None
        }

        dict_length_rel = {
            'OFC_STMT': None,
            'RES_ARTCL': None,
            'SOC_MED': None,
            'NEWS': None
        }

        dict_result_rel = {
            'OFC_STMT': None,
            'RES_ARTCL': None,
            'SOC_MED': None,
            'NEWS': None
        }

        # if target != 'OFC_STMT':
        #     print('OFC_STMT')
        #     dataset = pd.read_csv(_data_absolute_path + 'official_statements/data.csv')['content'].iloc[50:50+sample_size]
        #     dict_init['OFC_STMT'], dict_result['OFC_STMT'], dict_init_rel['OFC_STMT'], dict_result_rel['OFC_STMT'], dict_length['OFC_STMT'], dict_length_rel['OFC_STMT'] = adapt(dataset, 'OFC_STMT')
        if target != 'RES_ARTCL':
            print('RES_ARTCL')
            rootdir = _data_absolute_path + 'research_articles/document_parses/pdf_json'
            dataset = []
            for subdir, dirs, files in os.walk(rootdir):
                ix = 0
                files = files[50:50+sample_size]
                for f in files:
                    if ix == sample_size:
                        break
                    file = open(_data_absolute_path + 'research_articles/document_parses/pdf_json/' + f, 'r')
                    json_file = json.loads(str(file.read()))
                    research_article = ''
                    for paragraph in json_file['body_text']:
                        research_article = research_article + paragraph['text']

                    file.close()
                    dataset.append(research_article)
                    ix = ix + 1
            dict_init['RES_ARTCL'], dict_result['RES_ARTCL'], dict_init_rel['RES_ARTCL'], dict_result_rel['RES_ARTCL'], dict_length['RES_ARTCL'], dict_length_rel['RES_ARTCL'] = adapt(dataset, 'RES_ARTCL')
        #
        if target != 'SOC_MED':
            print('SOC_MED')
            dataset = pd.read_csv(_data_absolute_path + 'tweets/covid19_tweets.csv')['text'].iloc[(50 + iter*sample_size):(50 + (iter+1)*sample_size)]
            dict_init['SOC_MED'], dict_result['SOC_MED'], dict_init_rel['SOC_MED'], dict_result_rel['SOC_MED'], dict_length['SOC_MED'], dict_length_rel['SOC_MED'] = adapt(dataset, 'SOC_MED')
        # if target != 'NEWS':
        #     print('NEWS')
        #     dataset = pd.read_csv(_data_absolute_path + 'news/news.csv')['text'].iloc[(50 + iter*sample_size):(50 + (iter+1)*sample_size)]
        #     dict_init['NEWS'], dict_result['NEWS'], dict_init_rel['NEWS'], dict_result_rel['NEWS'], dict_length['NEWS'], dict_length_rel['NEWS'] = adapt(dataset, 'NEWS')

        keys = ['OFC_STMT', 'RES_ARTCL', 'SOC_MED', 'NEWS']
        keys.remove(target)

        # file = open(_results_absolute_path + "results_" + target + "_" + mode + ".txt", "a+")

        # for k in keys:
        #     print("Initial values for " + k, file=file)
        #     print(dict_init[k], file=file)
        #     print("Values after length manipulation for " + k, file=file)
        #     print(dict_length[k], file=file)
        #     print("Final values for " + k, file=file)
        #     print(dict_result[k], file=file)
        #     print("Initial relative_differences for " + k, file=file)
        #     print(dict_init_rel[k], file=file)
        #     print("Relative_differences after length manipulation for " + k, file=file)
        #     print(dict_length_rel[k], file=file)
        #     print("Final relative_differences for " + k, file=file)
        #     print(dict_result_rel[k], file=file)
        #
        # file.close()
