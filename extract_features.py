import argparse

import numpy
import torch

from src.excel_toolkit import get_sheet_tarr, get_feature_array
from src.helpers import SentEnc
from src.models import ClassificationModel, CEModel, FeatEnc
from src.test_cl import generate_cell_embeddings


def main(filename, sheet_name, ce_model_path, fe_model_path, cl_model_path, w2v_path, vocab_size, infersent_model):
    mode = 'ce+f'
    device = 'cpu'
    ce_dim = 512
    senc_dim = 4096
    window = 2
    f_dim = 43
    fenc_dim = 40
    n_classes = 6
    if device != 'cpu':
        torch.cuda.set_device(device)

    ce_model = CEModel(senc_dim, ce_dim // 2, window * 4)
    ce_model = ce_model.to(device)
    fe_model = FeatEnc(f_dim, fenc_dim)
    fe_model = fe_model.to(device)
    cl_model = ClassificationModel(ce_dim + fenc_dim, n_classes).to(device)

    ce_model.load_state_dict(torch.load(ce_model_path, map_location=device))
    fe_model.load_state_dict(torch.load(fe_model_path, map_location=device))
    cl_model.load_state_dict(torch.load(cl_model_path, map_location=device))

    print('loading word vectors...')
    senc = SentEnc(infersent_model, w2v_path, vocab_size, device=device, hp=False)

    tarr, n, m = get_sheet_tarr(filename, sheet_name, file_type='xlsx')
    # print(tarr)
    ftarr = get_feature_array(filename, sheet_name, file_type='xlsx')
    table = dict(table_array=tarr, feature_array=ftarr)

    sentences = set()
    for row in tarr:
        for c in row:
            sentences.add(c)
    senc.cache_sentences(list(sentences))

    # tarr_reshape = tarr.reshape(shape[0] * shape[1])
    # tarr_reshape[tarr_reshape == ''] = 'empty_string'
    # numpy.savetxt('metadata.tsv', tarr_reshape, delimiter='\t', fmt='%s')

    embeddings = generate_cell_embeddings(table, cl_model, ce_model, fe_model, senc, mode, device)
    print(embeddings.shape)
    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='processing inputs.')
    parser.add_argument('--file', type=str, help='path to the .xls spreadsheet.')
    parser.add_argument('--sheet', type=str, help='sheet name')
    parser.add_argument('--ce_model', type=str, help='path to the trained cell embedding model.')
    parser.add_argument('--fe_model', type=str, help='path to the trained feature encoding model.')
    parser.add_argument('--cl_model', type=str, help='path to the trained classification model.')
    parser.add_argument('--w2v', type=str, help='path to the glove embeddings.')
    parser.add_argument('--vocab_size', type=int, help='w2v vocab size.')
    parser.add_argument('--infersent_model', type=str, help='path to the infersent model.')
    parser.add_argument('--out', type=str, help='path to the output json file.')

    args = parser.parse_args()

    res = main(args.file, args.sheet, args.ce_model, args.fe_model, args.cl_model, args.w2v, args.vocab_size, args.infersent_model)
    # print(res.shape)
    # print(res)
