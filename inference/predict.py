import pandas as pd
from utils import *
from model import *
from transformers import T5EncoderModel, T5Tokenizer
from act_model import ActivityModel
from torch.utils.data import Dataset
from argparse import RawDescriptionHelpFormatter
import argparse

import csv

class EnzymeDatasets(Dataset):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)


def write_chunk_results(writer, ezy_keys, smiles_list, kcat_tensor, km_tensor, act_tensor):
    n_samples = len(ezy_keys)

    kcat_cpu = kcat_tensor.cpu().numpy()
    km_cpu = km_tensor.cpu().numpy()
    act_cpu = act_tensor.cpu().numpy()

    for i in range(n_samples):
        row = [
            ezy_keys[i],
            smiles_list[i],
            float(kcat_cpu[i, 0]),
            float(km_cpu[i, 0]),
            float(act_cpu[i, 0])
        ]
        writer.writerow(row)


if __name__ == "__main__":
    d = "RUN CATAPRO ..."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp_fpath", type=str, default="enzyme.fasta",
                         help="Input (.fasta). The path of enzyme file.")
    parser.add_argument("-model_dpath", type=str, default="model_dpah",
                         help="Input. The path of saved models.")
    parser.add_argument("-batch_size", type=int, default=64,
                        help="Input. Batch size")
    parser.add_argument("-embed_batch_size", type=int, default=16,
                        help="Input. Embedings batch size")
    parser.add_argument("-device", type=str, default="cuda",
                        help="Input. The device: cuda or cpu.")
    parser.add_argument("-out_fpath", type=str, default="catapro_predict_score.csv",
                        help="Input. Store the predicted kinetic parameters in this file..")
    args = parser.parse_args()

    inp_fpath = args.inp_fpath
    model_dpath = args.model_dpath
    batch_size = args.batch_size 
    device = args.device
    out_fpath = args.out_fpath
    embed_batch_size = args.embed_batch_size

    kcat_model_dpath = f"{model_dpath}/kcat_models"
    Km_model_dpath = f"{model_dpath}/Km_models"
    act_model_dpath = f"{model_dpath}/act_models"
    ProtT5_model = f"{model_dpath}/prot_t5_xl_uniref50/"
    MolT5_model = f"{model_dpath}/molt5-base-smiles2caption"

    prot_tokenizer = T5Tokenizer.from_pretrained(ProtT5_model, do_lower_case=False)
    prot_model = T5EncoderModel.from_pretrained(ProtT5_model)
    prot_model = prot_model.to(device).eval()

    mol_tokenizer = T5Tokenizer.from_pretrained(MolT5_model)
    mol_model = T5EncoderModel.from_pretrained(MolT5_model)
    mol_model = mol_model.to(device).eval()

    kcat_models, km_models, act_models = [], [], []
    for fold in range(10):
        kcat = KcatModel(device=device)
        kcat.load_state_dict(th.load(f"{kcat_model_dpath}/{fold}_bestmodel.pth", map_location=device))
        kcat.eval()
        kcat_models.append(kcat)

        km = KmModel(device=device)
        km.load_state_dict(th.load(f"{Km_model_dpath}/{fold}_bestmodel.pth", map_location=device))
        km.eval()
        km_models.append(km)

        act = ActivityModel(device=device)
        act.load_state_dict(th.load(f"{act_model_dpath}/{fold}_bestmodel.pth", map_location=device))
        act.eval()
        act_models.append(act)

    with open(out_fpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["fasta_id", "smiles", "pred_log10[kcat(s^-1)]",
                         "pred_log10[Km(mM)]", "pred_log10[kcat/Km(s^-1mM^-1)]"])

        total_processed = 0
        for chunk_df in pd.read_csv(inp_fpath, index_col=0, chunksize=batch_size):
            print(f"\n=== Processing chunk starting at row {total_processed} ===")

            ezy_ids = chunk_df["Enzyme_id"].values
            ezy_type = chunk_df["type"].values
            ezy_keys = [f"{_id}_{t}" for _id, t in zip(ezy_ids, ezy_type)]
            sequences = chunk_df["sequence"].values
            smiles = chunk_df["smiles"].values

            print(f"Generating embeddings for {len(sequences)} samples...")
            seq_embed = Seq_to_vec(sequences, prot_tokenizer, prot_model, device, embed_batch_size)
            mol_embed = get_molT5_embed(smiles, mol_tokenizer, mol_model, device, embed_batch_size)
            macc_embed = GetMACCSKeys(smiles, device)

            features = th.cat([seq_embed, mol_embed, macc_embed], dim=1).to(th.float32)

            print(f"Running predictions...")

            kcat_preds = th.zeros(len(sequences), 1, device=device)
            km_preds = th.zeros(len(sequences), 1, device=device)
            act_preds = th.zeros(len(sequences), 1, device=device)

            with th.no_grad():
                for fold in range(10):
                    ezy_feats = features[:, :1024]
                    sbt_feats = features[:, 1024:]

                    kcat_preds += kcat_models[fold](ezy_feats, sbt_feats).reshape(-1, 1)
                    km_preds += km_models[fold](ezy_feats, sbt_feats).reshape(-1, 1)
                    act_preds += act_models[fold](ezy_feats, sbt_feats)[-1].reshape(-1, 1)


            kcat_preds /= 10
            km_preds /= 10
            act_preds /= 10

            write_chunk_results(writer, ezy_keys, smiles,
                                kcat_preds, km_preds, act_preds)

            del seq_embed, mol_embed, macc_embed, features
            del kcat_preds, km_preds, act_preds
            th.cuda.empty_cache()

            total_processed += len(sequences)
            print(f"Completed {total_processed} samples total")
