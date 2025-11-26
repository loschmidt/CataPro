import torch
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys


def Seq_to_vec(sequences, tokenizer, model, device='cuda', batch_size=8):
    sequences_processed = [
        re.sub(r"[UZOB]", "X", ' '.join(
            list((seq[:500] + seq[-500:]) if len(seq) > 1000 else seq)))
        for seq in sequences]
    n_seqs = len(sequences_processed)
    features = torch.zeros((n_seqs, 1024), dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(0, n_seqs, batch_size):
            end_idx = min(i + batch_size, n_seqs)
            batch_seqs = sequences_processed[i:end_idx]

            ids = tokenizer.batch_encode_plus(
                batch_seqs, add_special_tokens=True,
                padding=True, return_tensors='pt'
            )
            input_ids = ids['input_ids'].to(device)
            attention_mask = ids['attention_mask'].to(device)

            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            torch.set_printoptions(linewidth=200, sci_mode=False, precision=9)

            for seq_num in range(len(batch_seqs)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                features[i + seq_num] = embedding.last_hidden_state[seq_num, :seq_len - 1, :].mean(dim=0).float()

    return features


def GetMACCSKeys(smiles_list, device='cuda'):
    N_smiles = len(smiles_list)
    features = torch.zeros((N_smiles, 167), dtype=torch.float32, device=device)

    if len(set(smiles_list)) == 1:
        mol = Chem.MolFromSmiles(smiles_list[0])
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_array = np.array([int(i) for i in fp.ToBitString()], dtype=np.float32)
        features[:] = torch.from_numpy(fp_array).to(device)
    else:
        for idx, smile in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smile)
            fp = MACCSkeys.GenMACCSKeys(mol)
            fp_array = np.array([int(i) for i in fp.ToBitString()], dtype=np.float32)
            features[idx] = torch.from_numpy(fp_array).to(device)
    return features


def get_molT5_embed(smiles_list, tokenizer, model, device='cuda', batch_size=16):
    N_smiles = len(smiles_list)
    features = torch.zeros((N_smiles, 768), dtype=torch.float32, device=device)

    if len(set(smiles_list)) == 1:
        # All identical
        with torch.no_grad():
            input_ids = tokenizer(smiles_list[0], return_tensors="pt").input_ids.to(device)
            outputs = model(input_ids=input_ids)
            embed = outputs.last_hidden_state[0, :-1, :].mean(dim=0).float()
            features[:] = embed
    else:
        with torch.no_grad():
            for i in range(0, N_smiles, batch_size):
                end_idx = min(i + batch_size, N_smiles)
                batch_smiles = smiles_list[i:end_idx]
                inputs = tokenizer.batch_encode_plus(
                    batch_smiles, add_special_tokens=True,
                    padding=True, return_tensors='pt'
                )
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                for j in range(len(batch_smiles)):
                    valid_len = attention_mask[j].sum() - 1
                    features[i + j] = outputs.last_hidden_state[j, :valid_len, :].mean(dim=0).float()
    torch.set_printoptions(precision=9)
    return features