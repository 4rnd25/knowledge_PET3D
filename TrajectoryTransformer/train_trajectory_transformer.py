"""
Created on Mar 05 2025 13:55

@author: ISAC - pettirsch
"""

import pandas as pd
import torch
import os
import random
from tqdm import tqdm

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from CommonUtils.ConfigUtils.read_config import load_config

from TrajectoryTransformer.Utils.dataloader import MultiFileTrajectoryDataset
from TrajectoryTransformer.Model.model import TrajectoryTransformer
from TrajectoryTransformer.Utils.fileListCreator import FileListCreator
from TrajectoryTransformer.Loss.loss import compute_loss
from TrajectoryTransformer.Utils.plot import plot_one_batch

import torch.multiprocessing as mp
from datetime import datetime

import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepend_mother_path(file_list, mother_path):
    return [os.path.join(mother_path, f.strip()) for f in file_list if f.strip()]

def validate_one_epoch(model, dataloader, epoch_num):
    model.eval()
    total_loss = 0
    total_rec_loss = 0
    total_cls_loss = 0
    counter = 0
    loop = tqdm(dataloader, desc="Training", total=len(dataloader), leave=False)

    with torch.no_grad():
        for masked_batch, full_batch, pad_mask, input_mask, cls_targets in dataloader:
            counter = counter + 1
            # if counter > 30:
            #     break
            reconstructed, cls_logits = model(masked_batch, pad_mask)

            loss, cls_loss, recon_loss = compute_loss(
                reconstructed,
                full_batch,
                cls_logits,
                cls_targets,
                pad_mask,  # only padding is excluded
                input_mask  # only masked positions are included
            )
            total_loss += loss.item()
            total_rec_loss += recon_loss.item()
            total_cls_loss += cls_loss.item()
            loop.update(1)
            loop.set_postfix(
                loss=total_loss / counter,
                iters=counter
            )

    return total_loss / len(dataloader), total_rec_loss / len(dataloader), total_cls_loss / len(dataloader)

def train_one_epoch(model, dataloader, optimizer, epoch_num):
    model.train()
    total_loss = 0
    total_rec_loss = 0
    total_cls_loss = 0
    counter = 0
    loop = tqdm(dataloader, desc="Training", total=len(dataloader), leave=False)
    for masked_batch, full_batch, pad_mask, input_mask, cls_targets in dataloader:
        counter += 1

        #if counter > 30:
        #    break
        optimizer.zero_grad()

        reconstructed, cls_logits = model(masked_batch, pad_mask)
        loss, cls_loss, recon_loss = compute_loss(
            reconstructed,
            full_batch,
            cls_logits,
            cls_targets,
            pad_mask,  # only padding is excluded
            input_mask  # only masked positions are included
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_rec_loss += recon_loss.item()
        total_cls_loss += cls_loss.item()
        loop.update(1)
        loop.set_postfix(
            loss=total_loss / counter,
            iters=counter
        )


    return total_loss / len(dataloader), total_rec_loss / len(dataloader), total_cls_loss / len(dataloader)

MAX_LEN = 1200  # or 1200, whatever your cap is

def collate_fn(batch):
    masked_tensors, full_tensors = [], []
    input_masks, lengths, classes = [], [], []

    for masked_df, full_df, cls_id in batch:
        L = len(masked_df)
        if L > MAX_LEN:
            # pick a stride so that ceil(L/stride) ≤ MAX_LEN
            stride = math.ceil(L / MAX_LEN)
            masked_df = masked_df.iloc[::stride]
            full_df   = full_df.iloc[::stride]
            # one more safety cut
            if len(masked_df) > MAX_LEN:
                masked_df = masked_df.iloc[:MAX_LEN]
                full_df   = full_df.iloc[:MAX_LEN]

        # now build your tensors exactly as before…
        m = torch.tensor(masked_df[['deltaT','deltaX','deltaY']].values, dtype=torch.float32)
        f = torch.tensor(full_df  [['deltaT','deltaX','deltaY']].values, dtype=torch.float32)

        mask_series = masked_df[['deltaX','deltaY']].isna().any(axis=1)
        inp_mask    = torch.tensor(mask_series.values, dtype=torch.bool)

        masked_tensors.append(torch.nan_to_num(m))
        full_tensors.append(f)
        input_masks.append(inp_mask)
        lengths.append(len(m))
        classes.append(cls_id)

    # …and your existing padding logic stays identical
    masked_batch = pad_sequence(masked_tensors, batch_first=True, padding_value=0.0).to(device)
    full_batch   = pad_sequence(full_tensors,   batch_first=True, padding_value=0.0).to(device)
    input_mask   = pad_sequence([t for t in input_masks], batch_first=True, padding_value=False).to(device)

    max_len      = masked_batch.size(1)
    lens         = torch.tensor(lengths, device=device).unsqueeze(1)
    idxs         = torch.arange(max_len, device=device).unsqueeze(0)
    pad_mask     = idxs >= lens

    class_tensor = torch.tensor(classes, dtype=torch.long, device=device)
    return masked_batch, full_batch, pad_mask, input_mask, class_tensor


def main_train_pipeline(config_lists, start_record_id_lists, end_record_id_lists, epochs=50, batch_size=16,
                        num_workers=4, verbose=False, mother_dataset=None):

    verbose = True

    # Check if file list exists
    file_list_train_path = "file_list_train.txt"
    file_list_val_path = "file_list_val.txt"
    if file_list_train_path and file_list_val_path:
        print("File lists already exist. Loading them.")
        with open(file_list_train_path, 'r') as f:
            file_list_train = [line.strip() for line in f.readlines()]
        with open(file_list_val_path, 'r') as f:
            file_list_val = [line.strip() for line in f.readlines()]
        if mother_dataset:
            file_list_train = prepend_mother_path(file_list_train, mother_dataset)
            file_list_val = prepend_mother_path(file_list_val, mother_dataset)
    else:
        # Create file lists
        file_list_creator = FileListCreator(mother_dataset, verbose)
        file_list_train = []
        file_list_val = []
        for config, start_rec_id, end_rec_id in zip(config_lists, start_record_id_lists, end_record_id_lists):
            config = load_config(config)
            file_list = file_list_creator.create_file_list(config, start_rec_id, end_rec_id)
            # Randomly select 20% of the files for validation
            num_files = len(file_list)
            num_val_files = max(num_files // 5, 1)
            val_indices = random.sample(range(num_files), num_val_files)
            file_list_train.extend([file_list[i] for i in range(num_files) if i not in val_indices])
            file_list_val.extend([file_list[i] for i in val_indices])
            if verbose:
                print(f"Training on {len(file_list_train)} files, validating on {len(file_list_val)} files for config {config}")

    file_list = file_list_train + file_list_val

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    # compose your directory name
    output_dir = f"{timestamp}_Training_Trajectory_Transformer"
    #output_dir = f"{timestamp}_Training_Trajectory_LSTM"

    # create the directory
    os.makedirs(output_dir, exist_ok=True)

    # SAve file list train and val
    with open(os.path.join(output_dir, "file_list_train.txt"), 'w') as f:
        for file in file_list_train:
            f.write(file + '\n')
    with open(os.path.join(output_dir, "file_list_val.txt"), 'w') as f:
        for file in file_list_val:
            f.write(file + '\n')

    # ── DATA SCALE CHECK (compute diffs from posXFit,posYFit) ───────────────────────
    total_dx = 0.0
    total_dy = 0.0
    total_n = 0
    for fpath in file_list_train:
        # read only the absolute positions
        df = pd.read_csv(fpath, usecols=['posXFit', 'posYFit'])

        # compute stepwise deltas (first row has no delta, so drop it)
        dx = df['posXFit'].diff().dropna().abs().sum()
        dy = df['posYFit'].diff().dropna().abs().sum()
        n = len(df) - 1
        total_dx += dx
        total_dy += dy
        total_n += n

    avg_dx = total_dx / total_n
    avg_dy = total_dy / total_n
    print(f"[DATA INFO] Avg |Δx| = {avg_dx:.4f},  Avg |Δy| = {avg_dy:.4f}")
    # ───────────────────────────────────────────────────────────────────────────────

    all_classes = set()
    for file in file_list:
        df = pd.read_csv(file, usecols=['ObjectClass'])
        all_classes.update(df['ObjectClass'].unique())

    class_map = {cls: i for i, cls in enumerate(sorted(all_classes))}

    train_dataset = MultiFileTrajectoryDataset(file_list_train, class_map, mask_ratio=0.4, noise_std=0.1)
    val_dataset = MultiFileTrajectoryDataset(file_list_val, class_map, mask_ratio=0.4, noise_std=0.1)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)

    # --- Model, Optimizer, Scheduler ---
    model = TrajectoryTransformer(n_classes=len(class_map)).to(device)
    #model = TrajectoryLSTM(n_classes=len(class_map)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    # generate YYYYMMDDTHHMMSS timestamp

    total_train_losses = []
    total_val_losses = []
    total_train_rec_losses = []
    total_train_cls_losses = []
    total_val_rec_losses = []
    total_val_cls_losses = []

    for epoch in range(epochs):
        train_loss, train_rec_loss, train_cls_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        val_loss, val_rec_loss, val_cls_loss = validate_one_epoch(model, val_loader, epoch)

        total_train_losses.append(train_loss)
        total_val_losses.append(val_loss)
        total_train_rec_losses.append(train_rec_loss)
        total_train_cls_losses.append(train_cls_loss)
        total_val_rec_losses.append(val_rec_loss)
        total_val_cls_losses.append(val_cls_loss)


        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

        # --- Save Visualization Plot ---
        plot_one_batch(train_loader, model, epoch+1, device, output_dir)

        # --- Save Best Model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved new best model (Val Loss: {val_loss:.4f})")

        # --- LR Scheduler Step ---
        scheduler.step(val_loss)

        # Save losses to CSV
        loss_df = pd.DataFrame({
            'epoch': range(1, epoch + 2),
            'train_loss': total_train_losses,
            'val_loss': total_val_losses,
            'train_rec_loss': total_train_rec_losses,
            'train_cls_loss': total_train_cls_losses,
            'val_rec_loss': total_val_rec_losses,
            'val_cls_loss': total_val_cls_losses
        })
        loss_csv_path = os.path.join(output_dir, "losses.csv")
        loss_df.to_csv(loss_csv_path, index=False)

        # Save plots of training and validation losses for each loss
        plt.figure(figsize=(12, 6))
        plt.plot(loss_df['epoch'], loss_df['train_loss'], label='Train Loss', color='blue')
        plt.plot(loss_df['epoch'], loss_df['val_loss'], label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "loss_plot.png"))
        plt.close()

        # Save plots of training and validation reconstruction losses
        plt.figure(figsize=(12, 6))
        plt.plot(loss_df['epoch'], loss_df['train_rec_loss'], label='Train Reconstruction Loss', color='green')
        plt.plot(loss_df['epoch'], loss_df['val_rec_loss'], label='Validation Reconstruction Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Training and Validation Reconstruction Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "rec_loss_plot.png"))
        plt.close()

        # Save plots of training and validation classification losses
        plt.figure(figsize=(12, 6))
        plt.plot(loss_df['epoch'], loss_df['train_cls_loss'], label='Train Classification Loss', color='purple')
        plt.plot(loss_df['epoch'], loss_df['val_cls_loss'], label='Validation Classification Loss', color='brown')
        plt.xlabel('Epoch')
        plt.ylabel('Classification Loss')
        plt.title('Training and Validation Classification Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "cls_loss_plot.png"))
        plt.close()



    # Optionally save final model
    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    config_lists = ['../Configs/Location_B.yaml',
                    '../Configs/Location_A.yaml']

    startRecIDList = [101310, 1]
    endRecIDList = [101444, 135] #101451

    mother_dataset = "/data/Arnd/Knowledge_PET3D_3D_Traffic_Trajectories"

    main_train_pipeline(config_lists, startRecIDList, endRecIDList, epochs=100 , batch_size=16, num_workers=16,
                        mother_dataset=mother_dataset) #16


