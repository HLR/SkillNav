import os
import json
import numpy as np
from transformers import AutoTokenizer

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:  
        if "/" not in split:    # the official splits
            if "NavNuances" in split:
                filepath = os.path.join(anno_dir, '%s_%s.json' % (dataset.upper(), split))
            elif tokenizer == 'bert':
                filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset.upper(), split))
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, '%s_%s_enc_xlmr.json' % (dataset.upper(), split))
            else:
                raise NotImplementedError('unspported tokenizer %s' % tokenizer)

            with open(filepath) as f:
                new_data = json.load(f)

            if split == 'val_train_seen':
                new_data = new_data[:50]

            # print(f"is_test: {is_test}")
            
            if not is_test:
                if dataset == 'r4r' and split == 'val_unseen':
                    ridxs = np.random.permutation(len(new_data))[:200]
                    new_data = [new_data[ridx] for ridx in ridxs]
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)
        # Join
        data += new_data
    return data

def load_instr_datasets_tok(anno_dir, dataset, splits, tokenizer, is_test=True):
    """ val_unseen val_numerical val_spatial """
    data = []
    for split in splits:
        filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset.upper(), split))

        with open(filepath) as f:
            new_data = json.load(f)
            for item in new_data:
                # compare_encodings = None
                # if "instr_encodings" in item.keys():
                #     compare_encodings = item["instr_encodings"]
                instr_encodings = []
                if 'instructions' in item:
                    for instruction in item["instructions"]:
                        # if 'instr_encodings' not in item:
                        token_ids = tokenizer(instruction)['input_ids']
                        instr_encodings.append(token_ids)
                else:
                    token_ids = tokenizer(item["instruction"])['input_ids']
                    instr_encodings.append(token_ids)
                item.update({"instr_encodings": instr_encodings})
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True, tokenizer_obj=None):
    if tokenizer_obj is not None:
        load_data = load_instr_datasets_tok(anno_dir, dataset, splits, tokenizer_obj, is_test=is_test)
    else:
        load_data = load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)

    data = []
    for i, item in enumerate(load_data):
        # Split multiple instructions into separate entries
        if 'instructions' in item:
            for j, instr in enumerate(item['instructions']):
                # if item['path_id'] != '224':
                #     continue
                
                new_item = dict(item)
                if len(item['instructions']) > 1:
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)  
                else:
                    new_item['instr_id'] = item['path_id']
                new_item['instruction'] = instr
                new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                del new_item['instructions']
                del new_item['instr_encodings']
                
                # print("-"*20)
                # print(new_item)
                    
                data.append(new_item)
        else:
            new_item = dict(item)
            # new_item['instr_id'] = item['path_id']
            new_item['instr_id'] = item['instr_id']
            new_item['instruction'] = item['instruction']
            if 'instr_encodings' in item:
                new_item['instr_encoding'] = item['instr_encodings'][0][:max_instr_len]
                del new_item['instr_encodings']
            else:
                new_item['instr_encoding'] = item['instr_encoding']
            data.append(new_item)

    return data

# def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True):
#     data = []
#     for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
#         # Split multiple instructions into separate entries
#         if 'instructions' in item:
#             for j, instr in enumerate(item['instructions']):
#                 new_item = dict(item)
#                 if len(item['instructions']) != 1:
#                     new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
#                 else:
#                     new_item['instr_id'] = item['path_id']
#                 new_item['instruction'] = instr
#                 if 'instr_encoding' in item:
#                     new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
#                     del new_item['instr_encodings']
#                 else:
#                     if tokenizer == 'bert':
#                         tokenizer_model = 'bert-base-uncased'
#                         tok = AutoTokenizer.from_pretrained(tokenizer_model)
#                     new_item['instr_encoding'] = tok.encode(instr)
#                 del new_item['instructions']
#                 data.append(new_item)
#         else:
#             instr = str(item['instruction'])
#             if tokenizer == 'bert':
#                 tokenizer_model = 'bert-base-uncased'
#                 tok = AutoTokenizer.from_pretrained(tokenizer_model)
#             item['instr_encoding'] = tok.encode(instr)
#             data.append(item)
            
#     return data

