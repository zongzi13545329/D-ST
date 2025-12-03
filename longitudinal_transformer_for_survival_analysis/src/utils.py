import os
import operator
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
import gc
from copy import deepcopy

### HELPER FUNCTIONS ###
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def val_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

### TIME-DEPENDENT EVALUATION METRICS FOLLOWING DYNAMIC-DEEPHIT ###
### CREDIT TO https://github.com/chl8856/Dynamic-DeepHit/blob/master/utils_eval.py ###
def C_index(Prediction, Time_survival, Death, Time):
    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
  
        if (Time_survival[i]<=Time and Death[i]==1):
            N_t[i,:] = 1

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result

def brier_score(Prediction, Time_survival, Death, Time):
    N = len(Prediction)
    y_true = ((Time_survival <= Time) * Death).astype(float)

    return np.mean((Prediction - y_true)**2)

### TRAINING/VALIDATION/INFERENCE LOOPS FOR LTSA ###
def train_LTSA(model, device, loss_fn, optimizer, data_loader, history, epoch, model_dir, amp, t_list, del_t_list, dataset, step_ahead=False):
    model.train()
    
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    running_loss = 0.
    patient_ids = []
    lateralities = []
    survs = []
    hazards = []
    censorships = []
    event_times = []
    obs_times = []
    for b, batch in pbar:
        optimizer.zero_grad()

        x = batch['x'].to(device)
        y = batch['y'].to(device)
        
        patient_id = batch['patient_id']
        laterality = batch['laterality']

        # For survival analysis
        censorship = batch['censorship'].to(device)
        event_time = batch['event_time']
        obs_time = batch['obs_time'].to(device)

        # For longitudinal analysis
        seq_length = batch['seq_length']
        rel_time = batch['rel_time']
        prior_AMD_sev = batch['prior_AMD_sev']

        with torch.autocast(enabled=amp, device_type='cuda', dtype=torch.float16):
            if step_ahead:
                hazard, surv, feat_pred, feat_target, padding_mask = model(x, seq_length, rel_time, prior_AMD_sev)
            else:
                hazard, surv, padding_mask = model(x, seq_length, rel_time, prior_AMD_sev)

            # Mask out padding tokens
            hazard_masked = torch.masked_select(hazard, padding_mask).reshape(-1, hazard.shape[-1])
            surv_masked = torch.masked_select(surv, padding_mask).reshape(-1, surv.shape[-1])
            censorship_masked = torch.masked_select(censorship, padding_mask.squeeze(-1))
            obs_time_masked = torch.masked_select(obs_time, padding_mask.squeeze(-1))
            y_masked = torch.masked_select(y, padding_mask.squeeze(-1))

            # Step-ahead feature prediction (if enabled)
            if step_ahead:
                # For step-ahead prediction, padding_mask has one extra 1 for each sequence than we want.
                # This is because the *last* element of the sequence has no "next" visit to predict features for.
                feat_padding_mask = torch.clone(padding_mask)
                for i in range(feat_padding_mask.shape[0]):
                    sum_ = feat_padding_mask[i, :].sum()
                    feat_padding_mask[i, sum_-1] = 0

                feat_pred_masked = torch.masked_select(feat_pred, feat_padding_mask).reshape(-1, feat_pred.shape[-1])

                feat_target_masked = torch.masked_select(feat_target, feat_padding_mask).reshape(-1, feat_target.shape[-1])

            # Compute primary survival loss
            if isinstance(loss_fn, list):
                loss = sum([loss_fn_(hazard_masked, surv_masked, y_masked, censorship_masked, obs_time_masked) for loss_fn_ in loss_fn])
            else:
                loss = loss_fn(hazard_masked, surv_masked, y_masked, censorship_masked, obs_time_masked)

            # Compute step-ahead prediction loss (if enabled)
            if step_ahead:
                mse_loss = torch.nn.functional.mse_loss(feat_pred_masked, feat_target_masked)
                loss += mse_loss

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        patient_ids.append(patient_id)
        lateralities.append(laterality)
        survs.append(surv_masked.detach().cpu().numpy())
        hazards.append(hazard_masked.detach().cpu().numpy())
        censorships.append(censorship_masked.detach().cpu().numpy())
        event_times.append(event_time)
        obs_times.append(obs_time_masked.detach().cpu().numpy())

        pbar.set_postfix({'loss': running_loss / (b + 1)})
        del x, y, hazard, surv, loss
        if isinstance(loss_fn, list):
            del loss_fn_
        torch.cuda.empty_cache()
    patient_ids = np.concatenate(patient_ids)
    lateralities = np.concatenate(lateralities)
    survs = np.concatenate(survs)
    censorships = np.concatenate(censorships)
    event_times = np.concatenate(event_times)
    obs_times = np.concatenate(obs_times)
    hazards = np.concatenate(hazards)

    # Organize eye-level event times and final observation times into DataFrame
    df = pd.DataFrame({'patient_id': patient_ids, 'laterality': lateralities, 'obs_time': obs_times, 'event_time': event_times})

    c_indices = []
    briers = []
    for t in t_list:
        sub_df = df[df['obs_time'] <= t]
        sub_df = sub_df.loc[sub_df.groupby(['patient_id', 'laterality'])['obs_time'].idxmax()]

        idx = sub_df.index.tolist()

        sub_censorships = censorships[idx]
        sub_event_times = event_times[idx]
        sub_survs = survs[idx]
        sub_obs_times = sub_df['obs_time'].values
        for del_t in del_t_list:
            # Compute risk of developing disease over window of interest [t, t + Δt]: P(t < T <= t + Δt | T > t) = (S(t) - S(t + Δt)) / S(t)
            if dataset == 'AREDS':
                # Let o_t = final observation time. We use 2*o_t to extract the proper survival probabilities since AREDS data was acquired in discrete 6-month intervals
                risks = np.array([(s[int(2*o_t)] - s[min(26, 2*(t+del_t))] + 1e-12) / (s[int(2*o_t)] + 1e-12) for o_t, s in zip(sub_obs_times, sub_survs)])
            else:
                risks = np.array([(s[int(o_t)] - s[min(14, t+del_t)] + 1e-12) / (s[int(o_t)] + 1e-12) for o_t, s in zip(sub_obs_times, sub_survs)])

            # Compute evaluation metrics
            c_index = C_index(risks, sub_event_times, (1-sub_censorships), t + del_t)
            print(f'[TRAIN] C-Index (t={t},del_t={del_t}) = {c_index:.3f}')

            brier = brier_score(risks, sub_event_times, (1-sub_censorships), t + del_t)
            print(f'[TRAIN] Brier (t={t},del_t={del_t}) = {brier:.3f}')

            c_indices.append(c_index)
            briers.append(brier)
    mean_c_index = np.mean(c_indices)
    mean_brier = np.mean(briers)
    print(f'[TRAIN] Mean C-Index: {mean_c_index:.3f}')
    print(f'[TRAIN] Mean Brier: {mean_brier:.3f}')

    # # Sanity check that predicted risks are higher for uncensored cases
    # idx = np.where(sub_censorships == 0)[0]
    # uncensored_risks = risks[idx]
    # censored_risks = risks[~idx]
    # print('---')
    # print(f'Predicted risks for disease (uncensored) cases: {uncensored_risks.mean():.3f} +/- {uncensored_risks.std():.3f}')
    # print(f'Predicted risks for censored cases: {censored_risks.mean():.3f} +/- {censored_risks.std():.3f}')
    # print('---')

    # Save validation metrics for this epoch
    current_metrics = pd.DataFrame([[epoch, 'train', running_loss / (b + 1)] + c_indices + briers + [mean_c_index, mean_brier]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    return pd.concat([history, current_metrics], axis=0)

def validate_LTSA(model, device, loss_fn, optimizer, scheduler, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, amp, t_list, del_t_list, dataset, step_ahead=False):
    model.eval()
    
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[VAL] Epoch {epoch}')

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    running_loss = 0.
    patient_ids = []
    lateralities = []
    survs = []
    hazards = []
    censorships = []
    event_times = []
    obs_times = []
    with torch.no_grad():
        for b, batch in pbar:

            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            patient_id = batch['patient_id']
            laterality = batch['laterality']

            # For survival analysis
            censorship = batch['censorship'].to(device)
            event_time = batch['event_time']
            obs_time = batch['obs_time'].to(device)

            # For longitudinal analysis
            seq_length = batch['seq_length']
            rel_time = batch['rel_time']
            prior_AMD_sev = batch['prior_AMD_sev']

            with torch.autocast(enabled=amp, device_type='cuda', dtype=torch.float16):
                if step_ahead:
                    hazard, surv, feat_pred, feat_target, padding_mask = model(x, seq_length, rel_time, prior_AMD_sev)
                else:
                    hazard, surv, padding_mask = model(x, seq_length, rel_time, prior_AMD_sev)

                # Mask out padding tokens
                hazard_masked = torch.masked_select(hazard, padding_mask).reshape(-1, hazard.shape[-1])
                surv_masked = torch.masked_select(surv, padding_mask).reshape(-1, surv.shape[-1])
                censorship_masked = torch.masked_select(censorship, padding_mask.squeeze(-1))
                obs_time_masked = torch.masked_select(obs_time, padding_mask.squeeze(-1))
                y_masked = torch.masked_select(y, padding_mask.squeeze(-1))

                # Step-ahead feature prediction (if enabled)
                if step_ahead:
                    # For step-ahead prediction, padding_mask has one extra 1 for each sequence than we want.
                    # This is because the *last* element of the sequence has no "next" visit to predict features for.
                    feat_padding_mask = torch.clone(padding_mask)
                    for i in range(feat_padding_mask.shape[0]):
                        sum_ = feat_padding_mask[i, :].sum()
                        feat_padding_mask[i, sum_-1] = 0

                    feat_pred_masked = torch.masked_select(feat_pred, feat_padding_mask).reshape(-1, feat_pred.shape[-1])

                    feat_target_masked = torch.masked_select(feat_target, feat_padding_mask).reshape(-1, feat_target.shape[-1])

                # Compute primary survival loss
                if isinstance(loss_fn, list):
                    loss = sum([loss_fn_(hazard_masked, surv_masked, y_masked, censorship_masked, obs_time_masked) for loss_fn_ in loss_fn])
                else:
                    loss = loss_fn(hazard_masked, surv_masked, y_masked, censorship_masked, obs_time_masked)

                # Compute step-ahead prediction loss (if enabled)
                if step_ahead:
                    mse_loss = torch.nn.functional.mse_loss(feat_pred_masked, feat_target_masked)
                    loss += mse_loss

            running_loss += loss.item()

            patient_ids.append(patient_id)
            lateralities.append(laterality)
            survs.append(surv_masked.detach().cpu().numpy())
            hazards.append(hazard_masked.detach().cpu().numpy())
            censorships.append(censorship_masked.detach().cpu().numpy())
            event_times.append(event_time)
            obs_times.append(obs_time_masked.detach().cpu().numpy())

            pbar.set_postfix({'loss': running_loss / (b + 1)})
    patient_ids = np.concatenate(patient_ids)
    lateralities = np.concatenate(lateralities)
    survs = np.concatenate(survs)
    censorships = np.concatenate(censorships)
    event_times = np.concatenate(event_times)
    obs_times = np.concatenate(obs_times)
    hazards = np.concatenate(hazards)

    # Organize eye-level event times and final observation times into DataFrame
    df = pd.DataFrame({'patient_id': patient_ids, 'laterality': lateralities, 'obs_time': obs_times, 'event_time': event_times})

    c_indices = []
    briers = []
    for t in t_list:
        sub_df = df[df['obs_time'] <= t]
        sub_df = sub_df.loc[sub_df.groupby(['patient_id', 'laterality'])['obs_time'].idxmax()]

        idx = sub_df.index.tolist()

        sub_censorships = censorships[idx]
        sub_event_times = event_times[idx]
        sub_survs = survs[idx]
        sub_obs_times = sub_df['obs_time'].values
        for del_t in del_t_list:
            # Compute risk of developing disease over window of interest [t, t + Δt]: P(t < T <= t + Δt | T > t) = (S(t) - S(t + Δt)) / S(t)
            if dataset == 'AREDS':
                # Let o_t = final observation time. We use 2*o_t to extract the proper survival probabilities since AREDS data was acquired in discrete 6-month intervals
                risks = np.array([(s[int(2*o_t)] - s[min(26, 2*(t+del_t))] + 1e-12) / (s[int(2*o_t)] + 1e-12) for o_t, s in zip(sub_obs_times, sub_survs)])
            else:
                risks = np.array([(s[int(o_t)] - s[min(14, t+del_t)] + 1e-12) / (s[int(o_t)] + 1e-12) for o_t, s in zip(sub_obs_times, sub_survs)])

            # Compute evaluation metrics
            c_index = C_index(risks, sub_event_times, (1-sub_censorships), t + del_t)
            print(f'[VAL] C-Index (t={t},del_t={del_t}) = {c_index:.3f}')

            brier = brier_score(risks, sub_event_times, (1-sub_censorships), t + del_t)
            print(f'[VAL] Brier (t={t},del_t={del_t}) = {brier:.3f}')

            c_indices.append(c_index)
            briers.append(brier)
    mean_c_index = np.mean(c_indices)
    mean_brier = np.mean(briers)
    print(f'[VAL] Mean C-Index: {mean_c_index:.3f}')
    print(f'[VAL] Mean Brier: {mean_brier:.3f}')

    # # Sanity check that predicted risks are higher for uncensored cases
    # idx = np.where(sub_censorships == 0)[0]
    # uncensored_risks = risks[idx]
    # censored_risks = risks[~idx]
    # print('---')
    # print(f'Predicted risks for disease (uncensored) cases: {uncensored_risks.mean():.3f} +/- {uncensored_risks.std():.3f}')
    # print(f'Predicted risks for censored cases: {censored_risks.mean():.3f} +/- {censored_risks.std():.3f}')
    # print('---')

    # Save validation metrics for this epoch
    current_metrics = pd.DataFrame([[epoch, 'val', running_loss / (b + 1)] + c_indices + briers + [mean_c_index, mean_brier]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    # Early stopping: save model weights only when validation metric has improved.
    if early_stopping_dict['metric'] == 'c-index':
        op = operator.gt
        current_metric = mean_c_index
    elif early_stopping_dict['metric'] == 'loss':
        op = operator.lt
        current_metric = running_loss / (b + 1)
    elif early_stopping_dict['metric'] == 'brier':
        op = operator.lt
        current_metric = mean_brier

    if scheduler is not None:
        scheduler.step(current_metric)

    if op(current_metric, early_stopping_dict['best_metric']):
        print(f'--- EARLY STOPPING: {early_stopping_dict["metric"].capitalize()} has improved from {early_stopping_dict["best_metric"]:.3f} to {current_metric:.3f}! Saving weights. ---')
        early_stopping_dict['epochs_no_improve'] = 0
        early_stopping_dict['best_metric'] = current_metric
        best_model_wts = deepcopy(model.state_dict())
        torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict(), 'epoch': epoch}, os.path.join(model_dir, f'best.pt'))
    else:
        print(f'--- EARLY STOPPING: {early_stopping_dict["metric"].capitalize()} has not improved from {early_stopping_dict["best_metric"]:.3f} ---')
        early_stopping_dict['epochs_no_improve'] += 1

    return pd.concat([history, current_metrics], axis=0), early_stopping_dict, best_model_wts

def evaluate_LTSA(model, device, loss_fn, data_loader, history, model_dir, weights, amp, t_list, del_t_list, dataset, step_ahead=False):
    model.eval()
    model.load_state_dict(weights, strict=True)
    
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[TEST] EVALUATION')

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    running_loss = 0.
    patient_ids = []
    lateralities = []
    survs = []
    hazards = []
    censorships = []
    event_times = []
    obs_times = []
    with torch.no_grad():
        for b, batch in pbar:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            patient_id = batch['patient_id']
            laterality = batch['laterality']

            # For survival analysis
            censorship = batch['censorship'].to(device)
            event_time = batch['event_time']
            obs_time = batch['obs_time'].to(device)

            # For longitudinal analysis
            seq_length = batch['seq_length']
            rel_time = batch['rel_time']
            prior_AMD_sev = batch['prior_AMD_sev']

            with torch.autocast(enabled=amp, device_type='cuda', dtype=torch.float16):
                if step_ahead:
                    hazard, surv, feat_pred, feat_target, padding_mask = model(x, seq_length, rel_time, prior_AMD_sev)
                else:
                    hazard, surv, padding_mask = model(x, seq_length, rel_time, prior_AMD_sev)

                # Mask out padding tokens
                hazard_masked = torch.masked_select(hazard, padding_mask).reshape(-1, hazard.shape[-1])
                surv_masked = torch.masked_select(surv, padding_mask).reshape(-1, surv.shape[-1])
                censorship_masked = torch.masked_select(censorship, padding_mask.squeeze(-1))
                obs_time_masked = torch.masked_select(obs_time, padding_mask.squeeze(-1))
                y_masked = torch.masked_select(y, padding_mask.squeeze(-1))

                # Step-ahead feature prediction (if enabled)
                if step_ahead:
                    # For step-ahead prediction, padding_mask has one extra 1 for each sequence than we want.
                    # This is because the *last* element of the sequence has no "next" visit to predict features for.
                    feat_padding_mask = torch.clone(padding_mask)
                    for i in range(feat_padding_mask.shape[0]):
                        sum_ = feat_padding_mask[i, :].sum()
                        feat_padding_mask[i, sum_-1] = 0

                    feat_pred_masked = torch.masked_select(feat_pred, feat_padding_mask).reshape(-1, feat_pred.shape[-1])

                    feat_target_masked = torch.masked_select(feat_target, feat_padding_mask).reshape(-1, feat_target.shape[-1])

                # Compute primary survival loss
                if isinstance(loss_fn, list):
                    loss = sum([loss_fn_(hazard_masked, surv_masked, y_masked, censorship_masked, obs_time_masked) for loss_fn_ in loss_fn])
                else:
                    loss = loss_fn(hazard_masked, surv_masked, y_masked, censorship_masked, obs_time_masked)

                # Compute step-ahead prediction loss (if enabled)
                if step_ahead:
                    mse_loss = torch.nn.functional.mse_loss(feat_pred_masked, feat_target_masked)
                    loss += mse_loss

            running_loss += loss.item()

            patient_ids.append(patient_id)
            lateralities.append(laterality)
            survs.append(surv_masked.detach().cpu().numpy())
            hazards.append(hazard_masked.detach().cpu().numpy())
            censorships.append(censorship_masked.detach().cpu().numpy())
            event_times.append(event_time)
            obs_times.append(obs_time_masked.detach().cpu().numpy())

            pbar.set_postfix({'loss': running_loss / (b + 1)})
    patient_ids = np.concatenate(patient_ids)
    lateralities = np.concatenate(lateralities)
    survs = np.concatenate(survs)
    censorships = np.concatenate(censorships)
    event_times = np.concatenate(event_times)
    obs_times = np.concatenate(obs_times)
    hazards = np.concatenate(hazards)

    # Organize eye-level event times and final observation times into DataFrame
    df = pd.DataFrame({'patient_id': patient_ids, 'laterality': lateralities, 'obs_time': obs_times, 'event_time': event_times})
    print(df)

    c_indices = []
    briers = []
    for t in t_list:
        sub_df = df[df['obs_time'] <= t]
        sub_df = sub_df.loc[sub_df.groupby(['patient_id', 'laterality'])['obs_time'].idxmax()]

        idx = sub_df.index.tolist()

        sub_censorships = censorships[idx]
        sub_event_times = event_times[idx]
        sub_survs = survs[idx]
        sub_obs_times = sub_df['obs_time'].values
        for del_t in del_t_list:
            # Compute risk of developing disease over window of interest [t, t + Δt]: P(t < T <= t + Δt | T > t) = (S(t) - S(t + Δt)) / S(t)
            if dataset == 'AREDS':
                # Let o_t = final observation time. We use 2*o_t to extract the proper survival probabilities since AREDS data was acquired in discrete 6-month intervals
                risks = np.array([(s[int(2*o_t)] - s[min(26, 2*(t+del_t))] + 1e-12) / (s[int(2*o_t)] + 1e-12) for o_t, s in zip(sub_obs_times, sub_survs)])
            else:
                risks = np.array([(s[int(o_t)] - s[min(14, t+del_t)] + 1e-12) / (s[int(o_t)] + 1e-12) for o_t, s in zip(sub_obs_times, sub_survs)])

            # Compute evaluation metrics
            c_index = C_index(risks, sub_event_times, (1-sub_censorships), t + del_t)
            print(f'[TEST] C-Index (t={t},del_t={del_t}) = {c_index:.3f}')

            brier = brier_score(risks, sub_event_times, (1-sub_censorships), t + del_t)
            print(f'[TEST] Brier (t={t},del_t={del_t}) = {brier:.3f}')

            c_indices.append(c_index)
            briers.append(brier)
    mean_c_index = np.mean(c_indices)
    mean_brier = np.mean(briers)
    print(f'[TEST] Mean C-Index: {mean_c_index:.3f}')
    print(f'[TEST] Mean Brier: {mean_brier:.3f}')

    # # Sanity check that predicted risks are higher for uncensored cases
    # idx = np.where(sub_censorships == 0)[0]
    # uncensored_risks = risks[idx]
    # censored_risks = risks[~idx]
    # print('---')
    # print(f'Predicted risks for disease (uncensored) cases: {uncensored_risks.mean():.3f} +/- {uncensored_risks.std():.3f}')
    # print(f'Predicted risks for censored cases: {censored_risks.mean():.3f} +/- {censored_risks.std():.3f}')
    # print('---')

    # Save predicted hazards for further evaluation
    out_pkl = []
    for i in range(hazards.shape[0]):
        inner_dict = {'patient_id': patient_ids[i], 'laterality': lateralities[i], 'obs_time': obs_times[i], 'hazards': hazards[i], 'survs': survs[i], 'censorship': censorships[i], 'event_time': event_times[i]}
        out_pkl.append(inner_dict)
    with open(os.path.join(model_dir, 'test_preds.pkl'), 'wb') as f:
        pickle.dump(out_pkl, f)

    # Plot training curves
    for metric in history.columns[2:]:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', metric], label='train')
        ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', metric], label='val')
        ax.set_xlabel('EPOCH')
        ax.set_ylabel(metric.upper())
        ax.legend()
        fig.savefig(os.path.join(model_dir, f'{metric}_history.png'), dpi=300, bbox_inches='tight')
    
    # Create + save table of C(t, Δt) values
    c_index_df = pd.DataFrame(index=[f't={t}' for t in t_list], columns=[[f'\u0394t={del_t}' for del_t in del_t_list]])
    i = 0
    for t in t_list:
        for del_t in del_t_list:
            c_index_df.loc[f't={t}', f'\u0394t={del_t}'] = c_indices[i]
            i += 1
    c_index_df.to_csv('test_c-index.csv')

    # Create + save table of C(t, Δt) values
    brier_df = pd.DataFrame(index=[f't={t}' for t in t_list], columns=[[f'\u0394t={del_t}' for del_t in del_t_list]])
    i = 0
    for t in t_list:
        for del_t in del_t_list:
            brier_df.loc[f't={t}', f'\u0394t={del_t}'] = briers[i]
            i += 1
    brier_df.to_csv('test_brier.csv')

    # Create + save text summary of testing results
    summary = f'Mean C-index: {mean_c_index:.3f}\n'
    summary += c_index_df.to_string() + '\n'
    summary += f'Brier score: {mean_brier:.3f}\n'
    summary += brier_df.to_string() + '\n'
    summary += f'Loss: {running_loss / (b+1)}'
    f = open(os.path.join(model_dir, f'test_summary.txt'), 'w')
    f.write(summary)
    f.close()

### TRAINING/VALIDATION/INFERENCE LOOPS FOR SINGLE-IMAGE BASELINE ###
def train(model, device, loss_fn, optimizer, data_loader, history, epoch, model_dir, amp, t_list, del_t_list, dataset, step_ahead=False):
    model.train()
    
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    running_loss = 0.
    patient_ids = []
    lateralities = []
    survs = []
    hazards = []
    censorships = []
    event_times = []
    obs_times = []
    for b, batch in pbar:
        optimizer.zero_grad()

        x = batch['x'].to(device)
        y = batch['y'].to(device)
        
        patient_id = batch['patient_id']
        laterality = batch['laterality']
        censorship = batch['censorship'].to(device)
        event_time = batch['event_time']
        obs_time = batch['obs_time']

        with torch.autocast(enabled=amp, device_type='cuda', dtype=torch.float16):
            hazard, surv = model(x)

            # Compute survival loss
            if isinstance(loss_fn, list):
                loss = sum([loss_fn_(hazard, surv, y, censorship, obs_time) for loss_fn_ in loss_fn])
            else:
                loss = loss_fn(hazard, surv, y, censorship, obs_time)

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        patient_ids.append(patient_id)
        lateralities.append(laterality)
        survs.append(surv.detach().cpu().numpy())
        hazards.append(hazard.detach().cpu().numpy())
        censorships.append(censorship.detach().cpu().numpy())
        event_times.append(event_time)
        obs_times.append(obs_time)

        pbar.set_postfix({'loss': running_loss / (b + 1)})
        del x, y, hazard, surv, loss
        if isinstance(loss_fn, list):
            del loss_fn_
        torch.cuda.empty_cache()
    gc.collect()
    patient_ids = np.concatenate(patient_ids)
    lateralities = np.concatenate(lateralities)
    survs = np.concatenate(survs)
    censorships = np.concatenate(censorships)
    event_times = np.concatenate(event_times)
    obs_times = np.concatenate(obs_times)
    hazards = np.concatenate(hazards)

    # Organize eye-level event times and final observation times into DataFrame
    df = pd.DataFrame({'patient_id': patient_ids, 'laterality': lateralities, 'obs_time': obs_times, 'event_time': event_times})

    c_indices = []
    briers = []
    for t in t_list:
        sub_df = df[df['obs_time'] <= t]
        sub_df = sub_df.loc[sub_df.groupby(['patient_id', 'laterality'])['obs_time'].idxmax()]

        idx = sub_df.index.tolist()

        sub_censorships = censorships[idx]
        sub_event_times = event_times[idx]
        sub_survs = survs[idx]
        sub_obs_times = sub_df['obs_time'].values
        for del_t in del_t_list:
            # Compute risk of developing disease over window of interest [t, t + Δt]: P(t < T <= t + Δt | T > t) = (S(t) - S(t + Δt)) / S(t)
            if dataset == 'AREDS':
                # Let o_t = final observation time. We use 2*o_t to extract the proper survival probabilities since AREDS data was acquired in discrete 6-month intervals
                risks = np.array([(s[int(2*o_t)] - s[min(26, 2*(t+del_t))] + 1e-12) / (s[int(2*o_t)] + 1e-12) for o_t, s in zip(sub_obs_times, sub_survs)])
            else:
                risks = np.array([(s[int(o_t)] - s[min(14, t+del_t)] + 1e-12) / (s[int(o_t)] + 1e-12) for o_t, s in zip(sub_obs_times, sub_survs)])

            # Compute evaluation metrics
            c_index = C_index(risks, sub_event_times, (1-sub_censorships), t + del_t)
            print(f'[TRAIN] C-Index (t={t},del_t={del_t}) = {c_index:.3f}')

            brier = brier_score(risks, sub_event_times, (1-sub_censorships), t + del_t)
            print(f'[TRAIN] Brier (t={t},del_t={del_t}) = {brier:.3f}')

            c_indices.append(c_index)
            briers.append(brier)
    mean_c_index = np.mean(c_indices)
    mean_brier = np.mean(briers)
    print(f'[TRAIN] Mean C-Index: {mean_c_index:.3f}')
    print(f'[TRAIN] Mean Brier: {mean_brier:.3f}')

    # # Sanity check that predicted risks are higher for uncensored cases
    # idx = np.where(sub_censorships == 0)[0]
    # uncensored_risks = risks[idx]
    # censored_risks = risks[~idx]
    # print(uncensored_risks.min(), uncensored_risks.max())
    # print('---')
    # print(f'Predicted risks for disease (uncensored) cases: {uncensored_risks.mean():.3f} +/- {uncensored_risks.std():.3f}')
    # print(f'Predicted risks for censored cases: {censored_risks.mean():.3f} +/- {censored_risks.std():.3f}')
    # print('---')

    # Save current training metrics for this epoch
    current_metrics = pd.DataFrame([[epoch, 'train', running_loss / (b + 1)] + c_indices + briers + [mean_c_index, mean_brier]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    return pd.concat([history, current_metrics], axis=0)

def validate(model, device, loss_fn, optimizer, scheduler, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, amp, t_list, del_t_list, dataset, step_ahead=False):
    model.eval()
    
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[VAL] Epoch {epoch}')

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    running_loss = 0.
    patient_ids = []
    lateralities = []
    survs = []
    hazards = []
    censorships = []
    event_times = []
    obs_times = []
    with torch.no_grad():
        for b, batch in pbar:
            optimizer.zero_grad()

            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            patient_id = batch['patient_id']
            laterality = batch['laterality']
            censorship = batch['censorship'].to(device)
            event_time = batch['event_time']
            obs_time = batch['obs_time']

            with torch.autocast(enabled=amp, device_type='cuda', dtype=torch.float16):
                hazard, surv = model(x)

                # Compute survival loss
                if isinstance(loss_fn, list):
                    loss = sum([loss_fn_(hazard, surv, y, censorship, obs_time) for loss_fn_ in loss_fn])
                else:
                    loss = loss_fn(hazard, surv, y, censorship, obs_time)
            running_loss += loss.item()

            patient_ids.append(patient_id)
            lateralities.append(laterality)
            survs.append(surv.detach().cpu().numpy())
            hazards.append(hazard.detach().cpu().numpy())
            censorships.append(censorship.detach().cpu().numpy())
            event_times.append(event_time)
            obs_times.append(obs_time)

            pbar.set_postfix({'loss': running_loss / (b + 1)})
    patient_ids = np.concatenate(patient_ids)
    lateralities = np.concatenate(lateralities)
    survs = np.concatenate(survs)
    censorships = np.concatenate(censorships)
    event_times = np.concatenate(event_times)
    obs_times = np.concatenate(obs_times)
    hazards = np.concatenate(hazards)

    # Organize eye-level event times and final observation times into DataFrame
    df = pd.DataFrame({'patient_id': patient_ids, 'laterality': lateralities, 'obs_time': obs_times, 'event_time': event_times})

    c_indices = []
    briers = []
    for t in t_list:
        sub_df = df[df['obs_time'] <= t]
        sub_df = sub_df.loc[sub_df.groupby(['patient_id', 'laterality'])['obs_time'].idxmax()]

        idx = sub_df.index.tolist()

        sub_censorships = censorships[idx]
        sub_event_times = event_times[idx]
        sub_survs = survs[idx]
        sub_obs_times = sub_df['obs_time'].values
        for del_t in del_t_list:
            # Compute risk of developing disease over window of interest [t, t + Δt]: P(t < T <= t + Δt | T > t) = (S(t) - S(t + Δt)) / S(t)
            if dataset == 'AREDS':
                # Let o_t = final observation time. We use 2*o_t to extract the proper survival probabilities since AREDS data was acquired in discrete 6-month intervals
                risks = np.array([(s[int(2*o_t)] - s[min(26, 2*(t+del_t))] + 1e-12) / (s[int(2*o_t)] + 1e-12) for o_t, s in zip(sub_obs_times, sub_survs)])
            else:
                risks = np.array([(s[int(o_t)] - s[min(14, t+del_t)] + 1e-12) / (s[int(o_t)] + 1e-12) for o_t, s in zip(sub_obs_times, sub_survs)])

            # Compute evaluation metrics
            c_index = C_index(risks, sub_event_times, (1-sub_censorships), t + del_t)
            print(f'[VAL] C-Index (t={t},del_t={del_t}) = {c_index:.3f}')

            brier = brier_score(risks, sub_event_times, (1-sub_censorships), t + del_t)
            print(f'[VAL] Brier (t={t},del_t={del_t}) = {brier:.3f}')

            c_indices.append(c_index)
            briers.append(brier)
    mean_c_index = np.mean(c_indices)
    mean_brier = np.mean(briers)
    print(f'[VAL] Mean C-Index: {mean_c_index:.3f}')
    print(f'[VAL] Mean Brier: {mean_brier:.3f}')

    # # Sanity check that predicted risks are higher for uncensored cases
    # idx = np.where(sub_censorships == 0)[0]
    # uncensored_risks = risks[idx]
    # censored_risks = risks[~idx]
    # print(uncensored_risks.min(), uncensored_risks.max())
    # print('---')
    # print(f'Predicted risks for disease (uncensored) cases: {uncensored_risks.mean():.3f} +/- {uncensored_risks.std():.3f}')
    # print(f'Predicted risks for censored cases: {censored_risks.mean():.3f} +/- {censored_risks.std():.3f}')
    # print('---')

    # Save current validation metrics for this epoch
    current_metrics = pd.DataFrame([[epoch, 'val', running_loss / (b + 1)] + c_indices + briers + [mean_c_index, mean_brier]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)


    # Early stopping: save model weights only when val (balanced) accuracy has improved
    import operator
    if early_stopping_dict['metric'] == 'c-index':
        op = operator.gt
        current_metric = mean_c_index
    elif early_stopping_dict['metric'] == 'loss':
        op = operator.lt
        current_metric = running_loss / (b + 1)
    elif early_stopping_dict['metric'] == 'brier':
        op = operator.lt
        current_metric = mean_brier
    if scheduler is not None:
        scheduler.step(current_metric)

    if op(current_metric, early_stopping_dict['best_metric']):
        print(f'--- EARLY STOPPING: {early_stopping_dict["metric"].capitalize()} has improved from {early_stopping_dict["best_metric"]:.3f} to {current_metric:.3f}! Saving weights. ---')
        early_stopping_dict['epochs_no_improve'] = 0
        early_stopping_dict['best_metric'] = current_metric
        best_model_wts = deepcopy(model.state_dict())
        torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict(), 'epoch': epoch}, os.path.join(model_dir, f'best.pt'))
    else:
        print(f'--- EARLY STOPPING: {early_stopping_dict["metric"].capitalize()} has not improved from {early_stopping_dict["best_metric"]:.3f} ---')
        early_stopping_dict['epochs_no_improve'] += 1

    return pd.concat([history, current_metrics], axis=0), early_stopping_dict, best_model_wts

def evaluate(model, device, loss_fn, data_loader, history, model_dir, weights, amp, t_list, del_t_list, dataset, step_ahead=False):
    model.eval()
    model.load_state_dict(weights, strict=True)
    
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[TEST] EVALUATION')

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    running_loss = 0.
    patient_ids = []
    lateralities = []
    survs = []
    hazards = []
    censorships = []
    event_times = []
    obs_times = []
    with torch.no_grad():
        for b, batch in pbar:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            patient_id = batch['patient_id']
            laterality = batch['laterality']
            censorship = batch['censorship'].to(device)
            event_time = batch['event_time']
            obs_time = batch['obs_time']

            with torch.autocast(enabled=amp, device_type='cuda', dtype=torch.float16):
                hazard, surv = model(x)

                # Compute primary survival loss
                if isinstance(loss_fn, list):
                    loss = sum([loss_fn_(hazard, surv, y, censorship, obs_time) for loss_fn_ in loss_fn])
                else:
                    loss = loss_fn(hazard, surv, y, censorship, obs_time)

            running_loss += loss.item()

            patient_ids.append(patient_id)
            lateralities.append(laterality)
            survs.append(surv.detach().cpu().numpy())
            hazards.append(hazard.detach().cpu().numpy())
            censorships.append(censorship.detach().cpu().numpy())
            event_times.append(event_time)
            obs_times.append(obs_time)

            pbar.set_postfix({'loss': running_loss / (b + 1)})
    patient_ids = np.concatenate(patient_ids)
    lateralities = np.concatenate(lateralities)
    survs = np.concatenate(survs)
    censorships = np.concatenate(censorships)
    event_times = np.concatenate(event_times)
    obs_times = np.concatenate(obs_times)
    hazards = np.concatenate(hazards)

    # Organize eye-level event times and final observation times into DataFrame
    df = pd.DataFrame({'patient_id': patient_ids, 'laterality': lateralities, 'obs_time': obs_times, 'event_time': event_times})

    c_indices = []
    briers = []
    for t in t_list:
        sub_df = df[df['obs_time'] <= t]
        sub_df = sub_df.loc[sub_df.groupby(['patient_id', 'laterality'])['obs_time'].idxmax()]

        idx = sub_df.index.tolist()

        sub_censorships = censorships[idx]
        sub_event_times = event_times[idx]
        sub_survs = survs[idx]
        sub_obs_times = sub_df['obs_time'].values
        for del_t in del_t_list:
            # Compute risk of developing disease over window of interest [t, t + Δt]: P(t < T <= t + Δt | T > t) = (S(t) - S(t + Δt)) / S(t)
            if dataset == 'AREDS':
                # Let o_t = final observation time. We use 2*o_t to extract the proper survival probabilities since AREDS data was acquired in discrete 6-month intervals
                risks = np.array([(s[int(2*o_t)] - s[min(26, 2*(t+del_t))] + 1e-12) / (s[int(2*o_t)] + 1e-12) for o_t, s in zip(sub_obs_times, sub_survs)])
            else:
                risks = np.array([(s[int(o_t)] - s[min(14, t+del_t)] + 1e-12) / (s[int(o_t)] + 1e-12) for o_t, s in zip(sub_obs_times, sub_survs)])

            # Compute evaluation metrics
            c_index = C_index(risks, sub_event_times, (1-sub_censorships), t + del_t)
            print(f'[TEST] C-Index (t={t},del_t={del_t}) = {c_index:.3f}')

            brier = brier_score(risks, sub_event_times, (1-sub_censorships), t + del_t)
            print(f'[TEST] Brier (t={t},del_t={del_t}) = {brier:.3f}')

            c_indices.append(c_index)
            briers.append(brier)
    mean_c_index = np.mean(c_indices)
    mean_brier = np.mean(briers)
    print(f'[TEST] Mean C-Index: {mean_c_index:.3f}')
    print(f'[TEST] Mean Brier: {mean_brier:.3f}')

    # # Sanity check that predicted risks are higher for uncensored cases
    # idx = np.where(sub_censorships == 0)[0]
    # uncensored_risks = risks[idx]
    # censored_risks = risks[~idx]
    # print(uncensored_risks.min(), uncensored_risks.max())
    # print('---')
    # print(f'Predicted risks for disease (uncensored) cases: {uncensored_risks.mean():.3f} +/- {uncensored_risks.std():.3f}')
    # print(f'Predicted risks for censored cases: {censored_risks.mean():.3f} +/- {censored_risks.std():.3f}')
    # print('---')

    # Save predicted hazards for further evaluation
    out_pkl = []
    for i in range(hazards.shape[0]):
        inner_dict = {'patient_id': patient_ids[i], 'laterality': lateralities[i], 'obs_time': obs_times[i], 'hazards': hazards[i], 'survs': survs[i], 'censorship': censorships[i], 'event_time': event_times[i]}
        out_pkl.append(inner_dict)
    with open(os.path.join(model_dir, 'test_preds.pkl'), 'wb') as f:
        pickle.dump(out_pkl, f)

    # Plot training curves
    for metric in history.columns[2:]:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', metric], label='train')
        ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', metric], label='val')
        ax.set_xlabel('EPOCH')
        ax.set_ylabel(metric.upper())
        ax.legend()
        fig.savefig(os.path.join(model_dir, f'{metric}_history.png'), dpi=300, bbox_inches='tight')
    
    # Create + save table of C(t, Δt) values
    c_index_df = pd.DataFrame(index=[f't={t}' for t in t_list], columns=[[f'\u0394t={del_t}' for del_t in del_t_list]])
    i = 0
    for t in t_list:
        for del_t in del_t_list:
            c_index_df.loc[f't={t}', f'\u0394t={del_t}'] = c_indices[i]
            i += 1
    c_index_df.to_csv('test_c-index.csv')

    # Create + save table of B(t, Δt) values
    brier_df = pd.DataFrame(index=[f't={t}' for t in t_list], columns=[[f'\u0394t={del_t}' for del_t in del_t_list]])
    i = 0
    for t in t_list:
        for del_t in del_t_list:
            brier_df.loc[f't={t}', f'\u0394t={del_t}'] = briers[i]
            i += 1
    brier_df.to_csv('test_brier.csv')

    # Create + save text summary of testing results
    summary = f'Mean C-index: {mean_c_index:.3f}\n'
    summary += c_index_df.to_string() + '\n'
    summary += f'Brier score: {mean_brier:.3f}\n'
    summary += brier_df.to_string() + '\n'
    summary += f'Loss: {running_loss / (b+1)}'
    f = open(os.path.join(model_dir, f'test_summary.txt'), 'w')
    f.write(summary)
    f.close()

### VFM-RELATED UTILS ###
# The following functions are adapted from the VisionFM codebase (src/VisionFM/utils.py)
# to support model loading and data normalization.

def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict.
    # From VisionFM/utils.py, originally from:
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    import math
    import logging
    _logger = logging.getLogger(__name__)
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def get_stats(modality):
    # From VisionFM/utils.py - Returns mean and std for different imaging modalities
    stats = {"MRI": [(0.17368862, 0.17368862, 0.17368862), (0.16198656, 0.16198656, 0.16198656)],
             "Fundus": [(0.423737496137619, 0.2609460651874542, 0.128403902053833),
                        (0.29482534527778625, 0.20167365670204163, 0.13668020069599152)],
             "UBM": [(0.096110865, 0.096110865, 0.096110865), (0.20977867, 0.20977867, 0.20977867)],
             "Ultrasound": [(0.05989236, 0.05978666, 0.056694973), (0.154277, 0.15504874, 0.14464583)],
             "External": [(0.4936253, 0.36324808, 0.25956994), (0.32001, 0.27109432, 0.21991591)],
             "FFA": [(0.2020487, 0.20205145, 0.20201068), (0.17406046, 0.17404206, 0.17401208)],
             "SlitLamp": [(0.5556667, 0.40288574, 0.38857886), (0.26516676, 0.23311588, 0.23903583)],
             "OCT": [(0.21091926, 0.21091926, 0.21091919), (0.17598894, 0.17598891, 0.17598893)]}
    assert modality in stats.keys(), f'unsupported modality: {modality}'
    return stats[modality]

def load_pretrained_weights(model, pretrained_weights, checkpoint_key=None, model_name=None, patch_size=None):
    # From VisionFM/utils.py - Loads pretrained weights with position embedding resizing
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        # position embedding
        if 'pos_embed' in state_dict and hasattr(model, 'pos_embed'):
            pos_embed_w = state_dict['pos_embed']
            if pos_embed_w.shape != model.pos_embed.shape:
                print(f"Will resize the pos_embed from {pos_embed_w.shape} to {model.pos_embed.shape}")
                pos_embed_w = resize_pos_embed(pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1),
                                            model.patch_embed.grid_size)
                state_dict['pos_embed'] = pos_embed_w

        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        return
    print("There is no reference weights available for this model => We use random weights.")
