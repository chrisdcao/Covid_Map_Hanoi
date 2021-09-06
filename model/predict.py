from modules.model import *
from transformers.tokenization_bert import BertTokenizer
from processor import NERProcessor, Example
from commons import NERdataset, FeatureExtractor
from torch.utils.data import DataLoader
from underthesea import sent_tokenize, word_tokenize

import torch
import torch.nn as nn
import logging
import argparse
import os
import time


class NER:

    """  Constructor cho class NER
            Args: 
                pretrain_dir: path to mô hình sử dung
                feat_dir: path to custom feature (mặc định sử dụng feature đc tích hợp sẵn)
                max_seq_length: độ dài câu tối đa mạng sẽ đọc trong 1 lần
                batch_size: dung lượng mỗi lần feed
                device: cpu hoặc gpu
    """
    def __init__(self, pretrain_dir="pretrains/baseline/models",
                 feat_dir=None,
                 max_seq_length=256,
                 batch_size=4,
                 device=torch.device('cpu')):

        #  Sử dụng Tokenize mặc định của BERT, chưa quan huấn luyện tokenize tiếng Việt, điều này khiến mô hình chưa tối ưu hoàn toàn
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        processor = NERProcessor(None, self.tokenizer)
        self.fe = FeatureExtractor(dict_dir=feat_dir) if feat_dir is not None else None
        self.label_list = processor.labels
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device
        num_labels = processor.get_num_labels()

        _, self.model, self.feature = model_builder_from_pretrained("bert-base-multilingual-cased",
                                                               num_labels,
                                                               pretrain_dir,
                                                               feat_dir=feat_dir)
        self.model.to(device)


    """  Chuyển đổi câu thành các feature (số chiều, để cho thành hệ số đầu ra tại embed layer)
            Input: 
                raw senetences
            Output: 
                features có shape (eid, tokens, token_ids, attention_masks, segment_ids, label_ids, label_masks, token_masks, feats)
    """
    def convert_sentences_to_features(self, sentences):
        features = []
        for sent_id, sentence in enumerate(sentences):
            if self.fe is None:
                words = " ".join(word_tokenize(sentence))
                ex_words = words.split()
            else:
                ex_words, ex_feats = self.fe.extract_feature(sentence)

            tokens = []
            feats = {}
            label_ids = []
            token_masks = []

            for i, word in enumerate(ex_words):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                for m in range(len(token)):
                    if m == 0:
                        token_masks.append(1)
                        if self.fe is not None:
                            for feat_key, feat_value in ex_feats[i]:
                                feat_id = self.feature.feature_infos[feat_key]['label'].index(feat_value) + 1
                                if feat_key not in feats:
                                    feats[feat_key] = [feat_id]
                                else:
                                    feats[feat_key].append(feat_id)
                    else:
                        token_masks.append(0)
                        if self.fe is not None:
                            for feat_key, _ in ex_feats[i]:
                                feats[feat_key].append(0)

            #  Nếu câu dài hơn độ dài sequence quy định, cắt đoạn đầu của câu để chiết xuất feature
            if len(tokens) >= self.max_seq_length - 1:
                tokens = tokens[0:(self.max_seq_length - 2)]
                token_masks = token_masks[0:(self.max_seq_length - 2)]
                for k, v in feats.items():
                    feats[k] = v[0:(self.max_seq_length - 2)]

            ntokens = []

            # Add [CLS] token
            ntokens.append("[CLS]")
            token_masks.insert(0, 0)

            if self.fe is not None:
                for feat_key, feat_value in self.feature.special_token["[CLS]"]:
                    feat_id = self.feature.feature_infos[feat_key]['label'].index(feat_value) + 1
                    feats[feat_key].insert(0, feat_id)

            ntokens.extend(tokens)

            # Add [SEP] token
            ntokens.append("[SEP]")
            token_masks.append(0)

            if self.fe is not None:
                for feat_key, feat_value in self.feature.special_token["[CLS]"]:
                    feat_id = self.feature.feature_infos[feat_key]['label'].index(feat_value) + 1
                    feats[feat_key].insert(0, feat_id)

            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            attention_masks = [1] * len(input_ids)
            label_masks = [1] * sum(token_masks)
            segment_ids = [0] * self.max_seq_length

            #  Pad những câu ko đủ độ dài quy chuẩn (256 mặc định) để có thể feed vào mạng (bởi độ dài khi vào mạng phải đồng nhất)
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids.extend(padding)
            attention_masks.extend(padding)
            token_masks.extend(padding)
            for k in feats.keys():
                feats[k].extend(padding)
            padding = [0] * (self.max_seq_length - len(label_masks))
            label_masks.extend(padding)

            assert len(input_ids) == self.max_seq_length
            assert len(attention_masks) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(label_masks) == self.max_seq_length
            assert len(token_masks) == self.max_seq_length
            assert sum(token_masks) == sum(label_masks)
            for k in feats.keys():
                assert len(feats[k]) == self.max_seq_length

            features.append(Example(eid=sent_id,
                                    tokens=" ".join(ex_words),
                                    token_ids=input_ids,
                                    attention_masks=attention_masks,
                                    segment_ids=segment_ids,
                                    label_ids=label_ids,
                                    label_masks=label_masks,
                                    token_masks=token_masks,
                                    feats=feats))
        return features


    """  Tiền xử lí: tách corpus (1 câu liền) thành các phân đoạn để sau đó vector hóa. Cách phân đoạn (tokenize) sẽ phụ thuộc vào mô hình tokenize đc sử dụng (hiện đang dùng loại mặc định cho đa ngôn ngữ của BERT).
            Input: 
                các corpus (ở dạng raw text)
            Output: 
                1 object DataLoader(data, batch_size)
    """
    def preprocess(self, text):
        sentences = sent_tokenize(text)
        features = self.convert_sentences_to_features(sentences)
        data = NERdataset(features, self.device)
        return DataLoader(data, batch_size=self.batch_size)


    """  Dự đoán và hậu xử lí
            Input: 
                raw text
            Output:
                list of tuple các địa điểm 
    """
    def predict(self, text):
        entites = []
        iterator = self.preprocess(text)
        for step, batch in enumerate(iterator):
            sents, token_ids, attention_masks, token_masks, segment_ids, label_ids, label_masks, feats = batch
            #  map probability để feed vào softmax
            logits = self.model(token_ids, attention_masks, token_masks, segment_ids, label_masks, feats)
            logits = torch.argmax(nn.functional.softmax(logits, dim=-1), dim=-1)
            pred = logits.detach().cpu().numpy()
            entity = None
            words = []
            for sent in sents:
                words.extend(sent.split())
            for p, w in list(zip(pred, words)):
                label = self.label_list[p-1]
                entity = (w, label)
                entites.append(entity)

        return entites


""" chương trình chạy
        Args: 
            pretrain_dir: đường dẫn tới mô hình chạy nhị phân (file *.bin)
            feat_dir: đường dẫn tới file feature tự chế (không sẽ sử dụng mặc định của mô hình)
            max_seq_len: độ dài tối đa của câu trong 1 xử lí 
            batch_size: độ to gói chia của đầu vào
            device: cpu hoặc gpu
        Input: 
            raw strings
        Output: 
            1. A txt_file containing locations, patient id, published date
            2. 1 text file contain last STT column for next-time crawl threshold
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_dir", default=None, type=str, required=True)
    parser.add_argument("--feat_dir", default=None, type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ner = NER(args.pretrain_dir, args.feat_dir, args.max_seq_length, args.batch_size, device)

    parent_dir = "covid_data/" 
    
    ##########################################################
    ##                                                      ##  
    ##    THAO TÁC TRÊN FILE_NAME CỦA CÁC TXT ĐÃ TẢI VỀ     ##
    ##                                                      ##  
    ##########################################################

    for file_name in os.listdir(parent_dir):

        if file_name.endswith('.txt'):
            i = 0
            #  Tách phần tên thành 3 phần bao gồm: Mã bệnh nhân - Quận ở - Ngày công bố
            chunks = file_name.split('%')
            #  Và lưu trữ vào các biến để về sau bot crawl sẽ lấy làm thông tin hiển thị
            #  case_id = chunks[0]
            patient_id = chunks[0]
            address_patient = chunks[1].replace('_',' ')
            date = chunks[2].split('.')[0].replace('_', '/') + '/' + time.strftime("%Y")

            #  print('\n------   Bắt đầu thông tin bệnh nhân {}  ------\n'.format(patient_id))

            print('Patient_ID: {}\nNgày công bố: {}\n'.format(patient_id, date))


            #####################################################
            ##                                                 ##  
            ##    ĐỌC NỘI DUNG FILE VÀ CHIẾT XUẤT ĐỊA ĐIỂM     ##
            ##                                                 ##  
            #####################################################
            
            with open(os.path.join(parent_dir, file_name), 'r') as f:
                input_text = f.read()
                locations = ner.predict(input_text)
                result = ''

                for index in range(0, len(locations)):
                    #  Hậu xử lí các tag đã thu thập được khi gặp phải tag 'O'
                    if locations[index][1] == 'O':
                        if locations[index][0] == ',' and result != '':
                            result = result.rstrip()
                            result += ', '
                        elif locations[index][0] != ',' and result != '':
                            result = result.rstrip()
                            if result[-1] == ',':
                                result = result[:-1]
                            if locations[index-1][1] != 'O':
                                if index < len(locations)-1 and locations[index+1][1] != 'O':
                                    result += ' ' + locations[index][0] + ' '
                                else:
                                    if address_patient in result and i == 0:

                                        if '.' not in result:
                                            #  sử dụng "%" như delimiter để lát sau split
                                            print('{}%nơi ở của bệnh nhân'.format(result))
                                            i += 1
                                    else:
                                        if '.' not in result:
                                            print('{}'.format(result))
                                    result = ''
                            else:
                                #  Đánh dấu nơi ở bệnh nhân trong các địa điểm tìm được
                                if address_patient in result and i == 0:
                                    if '.' not in result:
                                        print('{}%nơi ở của bệnh nhân'.format(result))
                                        i += 1
                                else:
                                    if '.' not in result:
                                        print('Address: {}'.format(result))
                                result = ''

                    #  Hợp địa điểm có cùng major tag (như hợp B-ORG vs I-ORG, B-PER  vs I-PER, etc..)
                    elif 'PER' in locations[index][1] or 'ORG' in locations[index][1] or 'LOC' in locations[index][1]:
                        result += locations[index][0] + ' '

            #  Hết 1 bệnh nhân thì xuống dòng nh lần để lát sau chúng ta dùng đc hàm split
            print('\n\n\n')
