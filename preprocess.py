
import re, numpy as np
import torch, pandas as pd
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from word2number import w2n

import en_core_sci_md
nlp_sci_md = en_core_sci_md.load()

import en_core_web_lg
nlp = en_core_web_lg.load()

input_path = "data/pubmedqa/pqal/train.tsv"
output_path = "data/pubmedqa/pqal/new_train.tsv"

tokenizer = BertTokenizer.from_pretrained('pretrained_weights/biobert_v1.1_pubmed', do_lower_case=False)

tokenizer.add_special_tokens({"additional_special_tokens":["[NUM]"]})

# tokenizer_large = BertTokenizer.from_pretrained('pretrained_weights/biobert_large', do_lower_case=False)

american_number_system = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        'thirteen': 13,
        'fourteen': 14,
        'fifteen': 15,
        'sixteen': 16,
        'seventeen': 17,
        'eighteen': 18,
        'nineteen': 19,
        'twenty': 20,
        'thirty': 30,
        'forty': 40,
        'fifty': 50,
        'sixty': 60,
        'seventy': 70,
        'eighty': 80,
        'ninety': 90,
        'hundred': 100,
        'thousand': 1000,
        'million': 1000000,
        'billion': 1000000000,
        'point': '.'
    }

def get_token_span(tokens, text):
#     token_spans = []
    start_offsets = []
    end_offsets = []
    for j in range(len(tokens)):
        if j == 0:
            token_offset = text.find(tokens[j])
        else:
            offset_begin = last_token_offset + len(tokens[j-1])
            text_to_find = text[offset_begin:]
            additional_offset = text_to_find.find(tokens[j])
            if additional_offset == -1 or additional_offset > len(tokens[j])*2:
                additional_offset = text_to_find.find(tokens[j][0])
            token_offset = offset_begin + additional_offset
#         token_spans.append((token_offset, token_offset+len(tokens[j])))
        start_offsets.append(token_offset)
        end_offsets.append(token_offset+len(tokens[j]))
        last_token_offset = token_offset
    return start_offsets, end_offsets

def isNumber(ent):
    number_sentence = ent.text.rstrip("s")
    number_sentence = number_sentence.replace('-', ' ')
    number_sentence = number_sentence.lower()  # converting input to lowercase

    if(number_sentence.isdigit()):  # return the number if user enters a number string
        return True, 

    split_words = number_sentence.strip().split()  # strip extra spaces and split sentence into words

    clean_numbers = []
    clean_decimal_numbers = []

    # removing and, & etc.
    for word in split_words:
        if word in american_number_system:
            clean_numbers.append(word)
        if word == "half" or word == "third":
            return False,

    # Error message if the user enters invalid input!
    if len(clean_numbers) == 0:
        return False, 
    elif split_words[-1] not in american_number_system:
        new_char = ent.end_char - len(split_words[-1]) - 1
        return True, new_char
    return True,
    

def get_ent_offsets(text):
    doc_sci = nlp_sci_md(text)
    start_offsets = [ent.start_char for ent in doc_sci.ents]
    end_offsets = [ent.end_char for ent in doc_sci.ents]
    return start_offsets, end_offsets


def get_new_text_num(text):
    num_positions, num_norm = get_number(text)
    last_end = 0
    new_text = ""
    if len(num_positions) == 0:
        return text, ""
    for num_pos in num_positions:
        new_text += text[last_end:num_pos[0]] + " [NUM] "
        last_end = num_pos[1]
    new_text += text[last_end:len(text)]
#     new_texts.append(new_text)
    input_ids = tokenizer.encode(new_text)
    nums = []
    nums = [in_id for in_id in input_ids if in_id == 1]
    if (len(nums) != len(num_positions)):
        print(new_text, num_norm)
    return new_text, " ".join(num_norm)

from dateutil.parser import parse

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except Exception as e:
        return False

def get_date_offsets(ents):
    starts = []
    ends = []
    for ent in ents:
        if ent.label_ == "DATE" and is_date(ent.text):
            starts.append(ent.start_char)
            ends.append(ent.end_char)
    return starts, ends

def get_number(text):
    num_positions = []
    num_norm = []
    matches = re.finditer("\d[\d,.]*", text)

    doc = nlp(text)
    ents= doc.ents
    ent_starts, ent_ends = get_ent_offsets(text)
    date_starts, date_ends = get_date_offsets(ents)
    i = 0
    skip = False
    for match in matches:
        nums = []
        
        s = match.start()
        e = match.end()
        if text[s:e].endswith('.') or text[s:e].endswith(','):
            e -= 1
        while (i < len(ents) and s > ents[i].start_char ):
            
            try:
                if ents[i].label_ in ["CARDINAL", "QUANTITY", "PERCENT"] and ents[i].end_char < s: 
                    isnumber = isNumber(ents[i])
                    if isnumber[0]:
                        if len(isnumber) == 1:
                            num_text = ents[i].text.replace(",", "")
                            num_positions.append([ents[i].start_char, ents[i].end_char])
                            num_norm.append(w2n.word_to_num(num_text))
                         
                        else:
#                             print(ents[i].text, ents[i].label_)
                            num_positions.append([ents[i].start_char, isnumber[1]])
                            num_norm.append(w2n.word_to_num(text[ents[i].start_char:isnumber[1]]))
                    
                i+=1
            except Exception as exception: 
                print(exception)
                print(ents[i].text, " ", ents[i].label_)
                i+=1
                continue
#             if (i < len(ents) and s > ents[i].start_char
#                 and e <= ents[i].end_char and ents[i].label_ == "PRODUCT"):
        if i < len(ents) and s == ents[i].start_char:
            i+=1
        if s > 0 and "." not in text[s:e] and text[s-1] == ".":
            s -= 1
        num_text = text[s:e].replace(",", "")
        
            
        try:
            number = float(num_text)
        except Exception as exception: 
            print(exception)
            continue
            
        if s > 0 and text[s-1].isalpha():
            continue
        
        if (len(ent_starts) > 0):
            ent_index = np.digitize(s, ent_starts) -1
            ent_text = text[ent_starts[ent_index]:ent_ends[ent_index]]
            if (e <= ent_ends[ent_index] and "=" not in ent_text and "<" not in ent_text):
                continue
                
        if (len(date_starts) > 0):
            date_index = np.digitize(s, date_starts) -1
            if e <= date_ends[date_index]:
                continue

        num_positions.append([s, e])
        num_norm.append(number)
    num_norm = [str(num) for num in num_norm]
    return num_positions, num_norm


reader = open(input_path, "r")

reader.readline()

writer = open(output_path, "w")

new_texts = []
while True:
    line = reader.readline()
    if not line:
        break
    split_lines = line.split("\t")
    
    index, query, context, label = split_lines[0], split_lines[1], split_lines[2], split_lines[3]
    new_query, query_nums = get_new_text_num(query)
    new_text, text_nums = get_new_text_num(context)
    writer.writelines("\t".join([ index, new_query, query_nums, new_text, text_nums, label]))
    
writer.close()
reader.close()
