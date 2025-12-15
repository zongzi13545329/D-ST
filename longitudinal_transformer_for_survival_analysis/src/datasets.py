import os

import albumentations as A
import albumentations.pytorch
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

### SINGLE-IMAGE DATASETS ###
class AREDS_Survival_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.split = split
        self.augment = augment
        self.tpe_mode = tpe_mode  # used for compatibility with longitudinal dataset
        self.learned_pe = learned_pe  # used for compatibility with longitudinal dataset
        
        self.label_df = pd.read_csv(os.path.join(self.label_dir, f'072123_{self.split}.csv'), dtype={'ID2': str, 'RC_ID': str})
        self.label_df['IMG'] = self.label_df['IMG'].apply(self._correct_fname)

        # Event time in discrete bin representing 6-month window from years 0-12
        self.label_df['label'] = self.label_df['time_to_event'].apply(lambda x: x*2).astype(int)

        self.valid_sample_count = 0  # Counter to track how many valid samples are used during training

        # Data augmentation pipeline
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 2), p=0.5),
                A.RandomResizedCrop(224, 224, scale=(0.5, 1), ratio=(0.75, 1.3333333333333333), p=0.5),
                A.Normalize(mean=(0.39293602, 0.21741964, 0.12034029), std=(0.3375143 , 0.19220693, 0.10813437)),  # computed from AREDS training set
                albumentations.pytorch.ToTensorV2(p=1)
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.39293602, 0.21741964, 0.12034029), std=(0.3375143 , 0.19220693, 0.10813437)),  # computed from AREDS training set
                albumentations.pytorch.ToTensorV2(p=1)
            ])

    def __len__(self):
        return self.label_df.shape[0]

    def _correct_fname(self, x):
        if x.endswith('.JPG'):
            x = x.replace('.JPG', '.jpg')

        return x

    def _load_image(self, f):
        x = cv2.imread(f)
        if x is None:
            raise FileNotFoundError(f"[Skipped] Failed to load image file: {f}")
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = self.transform(image=x)['image']
        return x

    def __getitem__(self, idx):
        max_retry = 10
        retry = 0

        while retry < max_retry:
            sample = self.label_df.iloc[idx]

            patient_id = sample['RC_ID']
            laterality = sample['LATERALITY']
            fname = sample['IMG']
            img_path = os.path.join(self.data_dir, patient_id, fname)

            try:
                x = self._load_image(img_path)
            except Exception as e:
                print(f"[Skipped] Failed to load image: {img_path}, reason: {e}")
                retry += 1
                idx = np.random.randint(0, len(self.label_df))
                continue

            label = sample['label']
            censorship = sample['censorship']
            event_time = sample['time_to_event']
            obs_time = sample['YEAR']

            y = np.array([label])

            self.valid_sample_count += 1

            return {
                'x': x,
                'y': torch.from_numpy(y).long(),
                'censorship': torch.from_numpy(np.array(censorship)).float(),
                'obs_time': obs_time,
                'event_time': event_time,
                'fname': fname,
                'patient_id': patient_id,
                'laterality': laterality
            }

        raise RuntimeError(f"Failed to load valid image after {max_retry} retries. Possibly all samples are corrupted.")

class OHTS_Survival_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.split = split
        self.augment = augment
        self.tpe_mode = tpe_mode  # used for compatibility with longitudinal dataset
        self.learned_pe = learned_pe  # used for compatibility with longitudinal dataset

        self.label_df = pd.read_csv(os.path.join(self.label_dir, f'clinic_split4_{self.split}.csv'))

        # Event time in discrete bin representing 1-year window from years 0-12
        self.label_df['label'] = self.label_df['time_to_event'].astype(int)
        self.valid_sample_count = 0

        # Data augmentation pipeline
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 2), p=0.5),
                A.RandomResizedCrop(224, 224, scale=(0.5, 1), ratio=(0.75, 1.3333333333333333), p=0.5),
                A.Normalize(mean=(0.52438926, 0.34920336, 0.21347666), std=(0.19437555, 0.12752805, 0.06420696)),  # Computed from OHTS training set
                albumentations.pytorch.ToTensorV2(p=1)
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.52438926, 0.34920336, 0.21347666), std=(0.19437555, 0.12752805, 0.06420696)),  # Computed from OHTS training set
                albumentations.pytorch.ToTensorV2(p=1)
            ])

    def __len__(self):
        return self.label_df.shape[0]

    def _correct_fname(self, x):
        if x.endswith('.JPG'):
            x = x.replace('.JPG', '.jpg')

        return x

    def _load_image(self, f):
        x = cv2.imread(f)
        if x is None:
            raise FileNotFoundError(f"[Skipped] Failed to load image file: {f}")
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return x

    def __getitem__(self, idx):
        max_retry = 10  # 最多尝试10次跳过坏样本
        retry = 0

        while retry < max_retry:
            sample = self.label_df.iloc[idx]

            # Get basic info for this eye
            patient_id = sample['ran']
            laterality = sample['lr']
            fname = f'{patient_id}-{sample["tracking"]}.jpg'
            img_path = os.path.join(self.data_dir, str(patient_id), fname)

            try:
                x = self._load_image(img_path)
            except Exception as e:
                print(f"[Skipped] Failed to load image: {img_path}, reason: {e}")
                retry += 1
                idx = np.random.randint(0, len(self.label_df))  # 换一个样本继续
                continue

            # Get survival-related info
            label = sample['label']
            censorship = sample['censorship']
            event_time = sample['time_to_event']
            obs_time = sample['year']

            if self.transform:
                x = self.transform(image=x)['image']

            y = np.array([label])
            self.valid_sample_count += 1

            return {
                'x': x,
                'y': torch.from_numpy(y).long(),
                'censorship': torch.from_numpy(np.array(censorship)).float(),
                'obs_time': obs_time,
                'event_time': event_time,
                'fname': fname,
                'patient_id': patient_id,
                'laterality': laterality
            }

        raise RuntimeError(f"Failed to load valid image after {max_retry} retries. Possibly all samples are corrupted.")
### LONGITUDINAL IMAGING DATASETS ###
class AREDS_Longitudinal_Survival_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.split = split
        self.augment = augment
        self.tpe_mode = tpe_mode
        self.learned_pe = learned_pe
        
        self.label_df = pd.read_csv(os.path.join(self.label_dir, f'072123_{self.split}.csv'), dtype={'ID2': str, 'RC_ID': str})
        self.label_df['IMG'] = self.label_df['IMG'].apply(self._correct_fname)

        # Event time in discrete bin representing 6-month window from years 0-12
        self.label_df['label'] = self.label_df['time_to_event'].apply(lambda x: x*2).astype(int)
        self.valid_sample_count = 0

        # Get info for unique eyes
        self.eye_dfs = self.label_df.groupby(['RC_ID', 'LATERALITY'])
        self.eye_ids = [i for i, _ in self.eye_dfs]

        # Set maximum sequence length (max. # observations per eye)
        self.max_seq_length = 14

        # Data augmentation pipeline
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 2), p=0.5),
                A.RandomResizedCrop(224, 224, scale=(0.5, 1), ratio=(0.75, 1.3333333333333333), p=0.5),
                A.Normalize(mean=(0.39293602, 0.21741964, 0.12034029), std=(0.3375143 , 0.19220693, 0.10813437)),  # computed from AREDS training set
                albumentations.pytorch.ToTensorV2(p=1)
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.39293602, 0.21741964, 0.12034029), std=(0.3375143 , 0.19220693, 0.10813437)),  # computed from AREDS training set
                albumentations.pytorch.ToTensorV2(p=1)
            ])

    def __len__(self):
        return len(self.eye_ids)

    def _correct_fname(self, x):
        if x.endswith('.JPG'):
            x = x.replace('.JPG', '.jpg')

        return x

    def _load_image(self, f):
        x = cv2.imread(f)
        if x is None:
            raise FileNotFoundError(f"[Skipped] Failed to load image file: {f}")
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = self.transform(image=x)['image']
        return x

    def __getitem__(self, idx):
        # Randomly sample an eye
        sample = self.eye_dfs.get_group(self.eye_ids[idx])

        # Get basic eye info
        patient_ids = sample['RC_ID'].values
        lateralities = sample['LATERALITY'].values
        fnames = sample['IMG'].values

        # Get survival-related eye info
        labels = sample['label'].values
        censorships = sample['censorship'].values
        event_times = sample['time_to_event'].values
        obs_times = sample['YEAR'].values

        # For learned temporal timestep encoding, use visit time in years
        if self.learned_pe:
            visit_times = obs_times
        else:
            # For temporal timestep encoding, can embed the visit time in discrete "bins" (6-month intervals) or months
            if self.tpe_mode == 'bins':
                visit_times = (obs_times*2).astype(int)
            elif self.tpe_mode == 'months':
                visit_times = (obs_times*12).astype(int)
            else:
                import sys
                sys.exit(-1)

        # Get AMD severity score from *previous* visit
        AMD_sevs = sample['AMDSEV'].values
        prior_AMD_sevs = np.zeros_like(AMD_sevs)
        if prior_AMD_sevs.size > 1:  # if more than one visit in sequence...
            prior_AMD_sevs[1:] = AMD_sevs[:-1]

        if np.all(np.isnan(prior_AMD_sevs)):
            # If entirely nan's, fill with zeros (pad everything)
            prior_AMD_sevs = np.zeros_like(prior_AMD_sevs)
        else:
            # Forward and backward fill NAs
            prior_AMD_sevs = pd.Series(prior_AMD_sevs).fillna(method='ffill').fillna(method='bfill').values

        # Load sequence of longitudinal images. x_seq: n_visits x 3 x 224 x 224
        x_seq = np.stack([self._load_image(os.path.join(self.data_dir, patient_id, fname)) for patient_id, fname in zip(patient_ids, fnames)], axis=0)
        seq_length = x_seq.shape[0]  # n_visits

        # Right-pad all relevant sequences with zeroes
        if x_seq.shape[0] < self.max_seq_length:
            x_seq = np.pad(x_seq, ((0, self.max_seq_length - x_seq.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
            visit_times = np.pad(visit_times, ((0, self.max_seq_length - visit_times.shape[0])), mode='constant')
            prior_AMD_sevs = np.pad(prior_AMD_sevs, ((0, self.max_seq_length - prior_AMD_sevs.shape[0])), mode='constant')
            labels = np.pad(labels, ((0, self.max_seq_length - labels.shape[0])), mode='constant')
            obs_times = np.pad(obs_times, ((0, self.max_seq_length - obs_times.shape[0])), mode='constant')
            censorships = np.pad(censorships, ((0, self.max_seq_length - censorships.shape[0])), mode='constant')
        
        # Event time label (same for each element in sequence)
        y = np.array(labels)
        self.valid_sample_count += 1

        return {'x': torch.from_numpy(x_seq).float(), 'seq_length': seq_length, 'y': torch.from_numpy(y).long(), 'fname': fnames, 'rel_time': torch.from_numpy(visit_times).long(), 'prior_AMD_sev': torch.from_numpy(prior_AMD_sevs).long(),
                'patient_id': patient_ids, 'laterality': lateralities, 'censorship': torch.from_numpy(np.array(censorships)).float(), 'event_time': event_times, 'obs_time': torch.from_numpy(obs_times).float()}

class OHTS_Longitudinal_Survival_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.split = split
        self.augment = augment
        self.tpe_mode = tpe_mode
        self.learned_pe = learned_pe
        
        self.label_df = pd.read_csv(os.path.join(self.label_dir, f'clinic_split3_{self.split}.csv'))
        # self.label_df = pd.read_csv(os.path.join(self.label_dir, f'072823_{self.split}.csv'))

        # Event time in discrete bin representing 1-year window from years 0-12
        self.label_df['label'] = self.label_df['time_to_event'].astype(int)
        self.valid_sample_count = 0

        # Get info for unique eyes
        self.eye_dfs = self.label_df.groupby(['ran', 'lr'])
        self.eye_ids = [i for i, _ in self.eye_dfs]

        # Set maximum sequence length
        self.max_seq_length = 14

        # Image augmentation pipeline
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 2), p=0.5),
                A.RandomResizedCrop(224, 224, scale=(0.5, 1), ratio=(0.75, 1.3333333333333333), p=0.5),
                A.Normalize(mean=(0.52438926, 0.34920336, 0.21347666), std=(0.19437555, 0.12752805, 0.06420696)),  # Computed from OHTS training set
                albumentations.pytorch.ToTensorV2(p=1)
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.52438926, 0.34920336, 0.21347666), std=(0.19437555, 0.12752805, 0.06420696)),  # Computed from OHTS training set
                albumentations.pytorch.ToTensorV2(p=1)
            ])

    def __len__(self):
        return len(self.eye_ids)

    def _load_image(self, f):
        x = cv2.imread(f)
        if x is None:
            raise FileNotFoundError(f"[Skipped] Failed to load image file: {f}")
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = self.transform(image=x)['image']
        return x

    def __getitem__(self, idx):
        # Randomly sample an eye
        sample = self.eye_dfs.get_group(self.eye_ids[idx])

        # Get basic eye info
        patient_ids = sample['ran'].values
        lateralities = sample['lr'].values
        trackings = sample['tracking'].values

        # Get survival-related eye info
        labels = sample['label'].values
        censorships = sample['censorship'].values
        event_times = sample['time_to_event'].values
        obs_times = sample['year'].values

        # For learned temporal timestep encoding, use visit time in years
        if self.learned_pe:
            visit_times = obs_times
        else:
            if self.tpe_mode == 'bins':
                visit_times = (obs_times * 2).astype(int)
            elif self.tpe_mode == 'months':
                visit_times = (obs_times * 12).astype(int)
            else:
                import sys
                sys.exit(-1)

        # Load sequence of longitudinal images
        x_seq = []
        valid_visit_times, valid_labels, valid_obs_times, valid_censorships = [], [], [], []
        valid_fnames = []

        for patient_id, tracking, vt, lbl, ot, c in zip(patient_ids, trackings, visit_times, labels, obs_times, censorships):
            fname = f"{patient_id}-{tracking}.jpg"
            img_path = os.path.join(self.data_dir, str(patient_id), fname)
            try:
                x = self._load_image(img_path)
                x_seq.append(x)
                valid_visit_times.append(vt)
                valid_labels.append(lbl)
                valid_obs_times.append(ot)
                valid_censorships.append(c)
                valid_fnames.append(fname)
            except Exception as e:
                print(f"[Skipped] Failed to load image: {img_path}, reason: {e}")
                continue

        if len(x_seq) == 0:
            raise RuntimeError(f"Failed to load valid image after {max_retry} retries. Possibly all samples are corrupted.")

        x_seq = np.stack(x_seq, axis=0)
        seq_length = x_seq.shape[0]

        # Pad everything if needed
        if seq_length < self.max_seq_length:
            pad_len = self.max_seq_length - seq_length
            x_seq = np.pad(x_seq, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')
            valid_visit_times = np.pad(valid_visit_times, (0, pad_len), mode='constant')
            valid_labels = np.pad(valid_labels, (0, pad_len), mode='constant')
            valid_obs_times = np.pad(valid_obs_times, (0, pad_len), mode='constant')
            valid_censorships = np.pad(valid_censorships, (0, pad_len), mode='constant')
            valid_fnames += [''] * pad_len  # pad missing filenames as empty string

        y = np.array(valid_labels)

        self.valid_sample_count += 1

        return {
            'x': torch.from_numpy(x_seq).float(),
            'seq_length': seq_length,
            'y': torch.from_numpy(y).long(),
            'fname': valid_fnames,
            'rel_time': torch.from_numpy(np.array(valid_visit_times)).long(),
            'prior_AMD_sev': None,
            'patient_id': patient_ids,
            'laterality': lateralities,
            'censorship': torch.from_numpy(np.array(valid_censorships)).float(),
            'event_time': event_times,
            'obs_time': torch.from_numpy(np.array(valid_obs_times)).float()
        }

