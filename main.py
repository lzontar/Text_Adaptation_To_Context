import json
import os
import random
import pandas as pd

import Common.library.Visualization as vis
import Common.classes.Text_Adaptation as ta
import Text_Characteristics

# text_characteristics = Text_Characteristics.Text_Characteristics(None, None)
# mean_measures = text_characteristics.mean_measures(True)
# variance_measures = text_characteristics.variance_map()
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
sample_size = 10
debug = False

initial_values = []
results = []
target_value = 30

origin_text_filename = './Data/Test/Input/test.txt'  # input("Origin text filename: ")
file = open(origin_text_filename, "r", encoding='utf8')
try:
    text = file.read()
except FileNotFoundError:
    print("File was not found. Please, input an existing file in the executing directory.")
    text = None
target = 'OFC_STMT'
tc = Text_Characteristics(target)
_data_absolute_path = 'C:/Luka/School/Bachelor/Bachelor\'s thesis/Text_Adaptation/Data/covid-19/'


def plus(dic1, dic2):
    return {
        "LEN": dic1["LEN"] + dic2["LEN"],
        "SENT_ANAL": {
            "POLAR": dic1["SENT_ANAL"]["POLAR"] + dic2["SENT_ANAL"]["POLAR"]
        },
        "READ": dic1["READ"] + dic2["READ"]
    }


def adapt(dataset, orig_type):
    sum_measures_init = {
            "LEN": 0,
            "SENT_ANAL": {
                "POLAR": 0
            },
            "READ": 0
        }
    sum_measures_result = {
            "LEN": 0,
            "SENT_ANAL": {
                "POLAR": 0
            },
            "READ": 0
        }

    count = 0
    for text in dataset:
        output = open("./Data/Test/Input/input_" + orig_type + str(count) + ".txt", "w")
        output.write(text)
        output.close()

        init_measures = tc.calc_text_measures(text)
        sum_measures_init = plus(sum_measures_init, init_measures)

        adapted_text = ta.adapt_text(text, mean_measures, target, debug)

        adapted_measures = tc.calc_text_measures(adapted_text)
        sum_measures_result = plus(sum_measures_result, adapted_measures)

        output = open("./Data/Test/Output/out_" + orig_type + str(count) + ".txt", "w")
        output.write(adapted_text)
        output.close()

        count = count + 1
    mean_measures_init = {
            "LEN": sum_measures_init["LEN"] / sample_size,
            "SENT_ANAL": {
                "POLAR": sum_measures_init["SENT_ANAL"]["POLAR"] / sample_size
            },
            "READ": sum_measures_init["READ"] / sample_size
        }

    mean_measures_init_rel = {
        "LEN": abs((mean_measures_init['LEN'] - mean_measures['LEN'][target]) / mean_measures['LEN'][target]),
        "SENT_ANAL": {
            "POLAR": abs((mean_measures_init['SENT_ANAL']['POLAR'] - mean_measures['SENT_ANAL'][target]['POLAR']) / mean_measures['SENT_ANAL'][target]['POLAR'])
        },
        "READ": abs((mean_measures_init['READ'] - mean_measures['READ'][target]) / mean_measures['READ'][target])
    }

    mean_measures_result = {
            "LEN": sum_measures_result["LEN"] / sample_size,
            "SENT_ANAL": {
                "POLAR": sum_measures_result["SENT_ANAL"]["POLAR"] / sample_size
            },
            "READ": sum_measures_result["READ"] / sample_size
        }

    mean_measures_result_rel = {
        "LEN": abs((mean_measures_result['LEN'] - mean_measures['LEN'][target]) / mean_measures['LEN'][target]),
        "SENT_ANAL": {
            "POLAR": abs((mean_measures_result['SENT_ANAL']['POLAR'] - mean_measures['SENT_ANAL'][target]['POLAR']) / mean_measures['SENT_ANAL'][target]['POLAR'])
        },
        "READ": abs((mean_measures_result['READ'] - mean_measures['READ'][target]) / mean_measures['READ'][target])
    }
    return mean_measures_init, mean_measures_result, mean_measures_init_rel, mean_measures_result_rel

dict_init = {
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

dict_result_rel = {
    'OFC_STMT': None,
    'RES_ARTCL': None,
    'SOC_MED': None,
    'NEWS': None
}

if target != 'OFC_STMT':
    dataset = random.sample(set(pd.read_csv(_data_absolute_path + 'official_statements/data.csv')['content']), sample_size)
    dict_init['OFC_STMT'], dict_result['OFC_STMT'], dict_init_rel['OFC_STMT'], dict_result_rel['OFC_STMT'] = adapt(dataset, 'OFC_STMT')
if target != 'RES_ARTCL':
    rootdir = _data_absolute_path + 'research_articles/document_parses/pdf_json'
    dataset = []
    for subdir, dirs, files in os.walk(rootdir):
        ix = 0
        files = random.sample(set(files), sample_size)
        for f in files:
            file = open(_data_absolute_path + 'research_articles/document_parses/pdf_json/' + f, 'r')
            json_file = json.loads(str(file.read()))
            research_article = ''
            for paragraph in json_file['body_text']:
                research_article = research_article + paragraph['text']

            file.close()
            dataset.append(research_article)
            ix = ix + 1
    dict_init['RES_ARTCL'], dict_result['RES_ARTCL'], dict_init_rel['RES_ARTCL'], dict_result_rel['RES_ARTCL'] = adapt(dataset, 'RES_ARTCL')

if target != 'SOC_MED':
    dataset = random.sample(set(pd.read_csv(_data_absolute_path + 'tweets/covid19_tweets.csv')['text']), sample_size)
    dict_init['SOC_MED'], dict_result['SOC_MED'], dict_init_rel['SOC_MED'], dict_result_rel['SOC_MED'] = adapt(dataset, 'SOC_MED')
if target != 'NEWS':
    dataset = random.sample(set(pd.read_csv(_data_absolute_path + 'news/news.csv')['text']), sample_size)
    dict_init['NEWS'], dict_result['NEWS'], dict_init_rel['NEWS'], dict_result_rel['NEWS'] = adapt(dataset, 'NEWS')

keys = ['OFC_STMT', 'RES_ARTCL', 'SOC_MED', 'NEWS']
keys.remove(target)

for k in keys:
    print("Initial values for " + k)
    print(dict_init)
    print("Final values for " + k)
    print(dict_result)
    print("Initial relative_differences for " + k)
    print(dict_init_rel)
    print("Final relative_differences for " + k)
    print(dict_result_rel)
