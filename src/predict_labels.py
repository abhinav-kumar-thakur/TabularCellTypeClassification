import argparse

import numpy as np
import torch

from src.excel_toolkit import get_sheet_tarr, get_feature_array, get_sheet_names
from src.helpers import SentEnc, label2ind
from src.models import ClassificationModel, CEModel, FeatEnc
from src.test_cl import predict_labels


class CellEmbeddingModelWrapper:
    def __init__(self, ce_model_path, fe_model_path, cl_model_path, w2v_path, vocab_size, infersent_model):
        self.mode = 'ce+f'
        self.device = 'cpu'
        self.ce_dim = 512
        self.senc_dim = 4096
        self.window = 2
        self.f_dim = 43
        self.fenc_dim = 40
        self.n_classes = 6
        if self.device != 'cpu': torch.cuda.set_device(self.device)

        self.ce_model = CEModel(self.senc_dim, self.ce_dim // 2, self.window * 4)
        self.ce_model = self.ce_model.to(self.device)
        self.fe_model = FeatEnc(self.f_dim, self.fenc_dim)
        self.fe_model = self.fe_model.to(self.device)
        self.cl_model = ClassificationModel(self.ce_dim + self.fenc_dim, self.n_classes).to(self.device)

        self.ce_model.load_state_dict(torch.load(ce_model_path, map_location=self.device))
        self.fe_model.load_state_dict(torch.load(fe_model_path, map_location=self.device))
        self.cl_model.load_state_dict(torch.load(cl_model_path, map_location=self.device))

        self.label2ind = ['attributes', 'data', 'header', 'metadata', 'derived', 'notes']

        print('loading word vectors...')
        self.senc = SentEnc(infersent_model, w2v_path, vocab_size, device=self.device, hp=False)

    def predict_labels(self, filename, sheet_names=None):
        result = dict()
        filetype = 'xlsx' if 'xlsx' in filename else 'xls'
        snames = sheet_names if sheet_names else get_sheet_names(filename, file_type=filetype)
        for sname in snames:
            print(f"Generating embedding for {filename} and {sname}")
            tarr, n, m = get_sheet_tarr(filename, sname, file_type=filetype)
            ftarr = get_feature_array(filename, sname, file_type=filetype)

            table = dict(table_array=tarr, feature_array=ftarr)

            sentences = set()
            for row in tarr:
                for c in row:
                    sentences.add(c)
            self.senc.cache_sentences(list(sentences))

            labels, probs, features = predict_labels(table, self.cl_model, self.ce_model, self.fe_model, self.senc, self.mode, self.device)
            probs = np.exp(probs)
            labels = np.vectorize(lambda x: label2ind[x])(labels)
            result[sname] = dict(table_arrays=tarr.tolist(), labels=labels.tolist(), labels_probs=probs.tolist(), embeddings=features)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='processing inputs.')
    parser.add_argument('--file', type=str,
                        help='path to the .xls spreadsheet.')
    parser.add_argument('--ce_model', type=str,
                        help='path to the trained cell embedding model.')
    parser.add_argument('--fe_model', type=str,
                        help='path to the trained feature encoding model.')
    parser.add_argument('--cl_model', type=str,
                        help='path to the trained classification model.')
    parser.add_argument('--w2v', type=str,
                        help='path to the glove embeddings.')
    parser.add_argument('--vocab_size', type=int,
                        help='w2v vocab size.')
    parser.add_argument('--infersent_model', type=str,
                        help='path to the infersent model.')
    parser.add_argument('--infersent_source', type=str,
                        help='path to the infersent source code.')
    parser.add_argument('--out', type=str,
                        help='path to the output json file.')

    args = parser.parse_args()

    # res = main(args.file, args.ce_model, args.fe_model, args.cl_model, args.w2v, args.vocab_size, args.infersent_source, args.infersent_model)
    # json.dump(res, open(args.out, 'w'))
