import os

import albumentations as A
import albumentations.pytorch
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

# 使用绝对导入
from src.utils import get_stats

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

        self.label_df = pd.read_csv(os.path.join(self.label_dir, f'072823_{self.split}.csv'))

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
        
        self.label_df = pd.read_csv(os.path.join(self.label_dir, f'072823_{self.split}.csv'))

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

### VFM-SPECIFIC DATASETS ###
class VFM_AREDS_Survival_Dataset(AREDS_Survival_Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        # Call parent constructor first
        super().__init__(data_dir, label_dir, split, augment, tpe_mode, learned_pe)
        
        # Get VFM's normalization values for Fundus images
        vfm_mean, vfm_std = get_stats('Fundus')
        
        # Override the transform to use VFM's normalization values
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 2), p=0.5),
                A.RandomResizedCrop(224, 224, scale=(0.5, 1), ratio=(0.75, 1.3333333333333333), p=0.5),
                A.Normalize(mean=vfm_mean, std=vfm_std),  # VFM's normalization for Fundus images
                albumentations.pytorch.ToTensorV2(p=1)
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=vfm_mean, std=vfm_std),  # VFM's normalization for Fundus images
                albumentations.pytorch.ToTensorV2(p=1)
            ])

class VFM_OHTS_Survival_Dataset(OHTS_Survival_Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        # Call parent constructor first
        super().__init__(data_dir, label_dir, split, augment, tpe_mode, learned_pe)
        
        # Get VFM's normalization values for Fundus images
        vfm_mean, vfm_std = get_stats('Fundus')
        
        # Override the transform to use VFM's normalization values
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 2), p=0.5),
                A.RandomResizedCrop(224, 224, scale=(0.5, 1), ratio=(0.75, 1.3333333333333333), p=0.5),
                A.Normalize(mean=vfm_mean, std=vfm_std),  # VFM's normalization for Fundus images
                albumentations.pytorch.ToTensorV2(p=1)
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=vfm_mean, std=vfm_std),  # VFM's normalization for Fundus images
                albumentations.pytorch.ToTensorV2(p=1)
            ])

class VFM_AREDS_Longitudinal_Survival_Dataset(AREDS_Longitudinal_Survival_Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        # Call parent constructor first
        super().__init__(data_dir, label_dir, split, augment, tpe_mode, learned_pe)
        
        # Get VFM's normalization values for Fundus images
        vfm_mean, vfm_std = get_stats('Fundus')
        
        # Override the transform to use VFM's normalization values
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 2), p=0.5),
                A.RandomResizedCrop(224, 224, scale=(0.5, 1), ratio=(0.75, 1.3333333333333333), p=0.5),
                A.Normalize(mean=vfm_mean, std=vfm_std),  # VFM's normalization for Fundus images
                albumentations.pytorch.ToTensorV2(p=1)
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=vfm_mean, std=vfm_std),  # VFM's normalization for Fundus images
                albumentations.pytorch.ToTensorV2(p=1)
            ])

class VFM_OHTS_Longitudinal_Survival_Dataset(OHTS_Longitudinal_Survival_Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        # Call parent constructor first
        super().__init__(data_dir, label_dir, split, augment, tpe_mode, learned_pe)
        
        # Get VFM's normalization values for Fundus images
        vfm_mean, vfm_std = get_stats('Fundus')
        
        # Override the transform to use VFM's normalization values
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 2), p=0.5),
                A.RandomResizedCrop(224, 224, scale=(0.5, 1), ratio=(0.75, 1.3333333333333333), p=0.5),
                A.Normalize(mean=vfm_mean, std=vfm_std),  # VFM's normalization for Fundus images
                albumentations.pytorch.ToTensorV2(p=1)
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=vfm_mean, std=vfm_std),  # VFM's normalization for Fundus images
                albumentations.pytorch.ToTensorV2(p=1)
            ])

### SIGF DATASETS ###
class SIGF_Survival_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.split = split
        self.augment = augment
        self.tpe_mode = tpe_mode
        self.learned_pe = learned_pe
        
        # Load CSV file - SIGF uses simple names: train.csv, test.csv, validation.csv
        csv_name = 'validation.csv' if split == 'val' else f'{split}.csv'
        self.label_df = pd.read_csv(os.path.join(self.label_dir, csv_name))
        
        # Use time_to_event as label (consistent with AREDS/OHTS)
        # time_to_event = total sequence length for each patient-eye (e.g., 3 for SD1294)
        # year = current observation index (0, 1, 2 for SD1294's 3 images)
        # For survival analysis, we predict time_to_event, not current observation time
        self.label_df['label'] = self.label_df['time_to_event'].astype(int)
        self.valid_sample_count = 0
        
        # SIGF preprocessing - EXACT DeepGF compatibility
        # NO augmentation regardless of augment parameter to maintain DeepGF compatibility
        self.transform = A.Compose([
            A.Resize(224, 224),  # DeepGF: image.resize((224, 224))
            # DeepGF normalization: image = image / 255.0 (simple 0-1 normalization)
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            albumentations.pytorch.ToTensorV2(p=1)
        ])
    
    def __len__(self):
        return self.label_df.shape[0]
    
    def _get_image_path(self, patient_id, laterality, split):
        """Get image directory path based on split"""
        patient_eye_id = f"{patient_id}_{laterality}"
        if split == 'train':
            return os.path.join(self.data_dir, 'train', 'image', 'all', 'image', patient_eye_id)
        else:
            # Map 'val' to 'validation' for directory name
            dir_name = 'validation' if split == 'val' else split
            return os.path.join(self.data_dir, dir_name, 'image', patient_eye_id)
    
    def _find_image_file(self, img_dir, patient_id, laterality, year_idx):
        """Find the actual image file for a given patient and time point"""
        # List all files in the directory
        try:
            files = os.listdir(img_dir)
            # Filter files that match the patient and laterality (handle both regular and _01 suffix files)
            pattern_files = [f for f in files if f.startswith(f"{patient_id}_") and (
                f.endswith(f"_{laterality}.JPG") or f.endswith(f"_{laterality}.jpg") or
                f.endswith(f"_{laterality}_01.JPG") or f.endswith(f"_{laterality}_01.jpg")
            )]
            
            if len(pattern_files) == 0:
                raise FileNotFoundError(f"No image files found for {patient_id}_{laterality}")
            
            # Sort files by date (assuming YYYY_MM_DD format)
            pattern_files.sort()
            
            # Return the file at the specified index
            if year_idx < len(pattern_files):
                return pattern_files[year_idx]
            else:
                raise IndexError(f"Year index {year_idx} out of range for {patient_id}_{laterality}")
        except Exception as e:
            raise FileNotFoundError(f"Error finding image file: {e}")
    
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
            
            patient_id = sample['ID']
            laterality = sample['LR']
            year_idx = sample['year']
            
            # Get image directory
            img_dir = self._get_image_path(patient_id, laterality, self.split)
            
            try:
                # Find and load the image
                fname = self._find_image_file(img_dir, patient_id, laterality, year_idx)
                img_path = os.path.join(img_dir, fname)
                x = self._load_image(img_path)
            except Exception as e:
                print(f"[Skipped] Failed to load image: {patient_id}_{laterality}_{year_idx}, reason: {e}")
                retry += 1
                idx = np.random.randint(0, len(self.label_df))
                continue
            
            label = sample['label']
            censorship = sample['censorship']
            event_time = sample['time_to_event']
            obs_time = sample['year']
            
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

class SIGF_Longitudinal_Survival_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.split = split
        self.augment = augment
        self.tpe_mode = tpe_mode
        self.learned_pe = learned_pe
        
        # Load CSV file
        csv_name = 'validation.csv' if split == 'val' else f'{split}.csv'
        self.label_df = pd.read_csv(os.path.join(self.label_dir, csv_name))
        
        # Use time_to_event as label (consistent with AREDS/OHTS)
        # time_to_event = total sequence length for each patient-eye (e.g., 3 for SD1294)
        # year = current observation index (0, 1, 2 for SD1294's 3 images)
        # For survival analysis, we predict time_to_event, not current observation time
        self.label_df['label'] = self.label_df['time_to_event'].astype(int)
        self.valid_sample_count = 0
        
        # Group by patient-eye combinations
        self.eye_dfs = self.label_df.groupby(['ID', 'LR'])
        self.eye_ids = [i for i, _ in self.eye_dfs]
        
        # Set maximum sequence length to match framework requirement
        self.max_seq_length = 14
        
        # SIGF preprocessing - EXACT DeepGF compatibility
        # NO augmentation regardless of augment parameter to maintain DeepGF compatibility
        self.transform = A.Compose([
            A.Resize(224, 224),  # DeepGF: image.resize((224, 224))
            # DeepGF normalization: image = image / 255.0 (simple 0-1 normalization)
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            albumentations.pytorch.ToTensorV2(p=1)
        ])
    
    def __len__(self):
        return len(self.eye_ids)
    
    def _get_image_path(self, patient_id, laterality, split):
        """Get image directory path based on split"""
        patient_eye_id = f"{patient_id}_{laterality}"
        if split == 'train':
            return os.path.join(self.data_dir, 'train', 'image', 'all', 'image', patient_eye_id)
        else:
            # Map 'val' to 'validation' for directory name
            dir_name = 'validation' if split == 'val' else split
            return os.path.join(self.data_dir, dir_name, 'image', patient_eye_id)
    
    def _find_image_files(self, img_dir, patient_id, laterality):
        """Find all image files for a given patient-eye combination"""
        try:
            files = os.listdir(img_dir)
            # Filter files that match the patient and laterality (handle both regular and _01 suffix files)
            pattern_files = [f for f in files if f.startswith(f"{patient_id}_") and (
                f.endswith(f"_{laterality}.JPG") or f.endswith(f"_{laterality}.jpg") or
                f.endswith(f"_{laterality}_01.JPG") or f.endswith(f"_{laterality}_01.jpg")
            )]
            
            if len(pattern_files) == 0:
                raise FileNotFoundError(f"No image files found for {patient_id}_{laterality}")
            
            # Sort files by date (assuming YYYY_MM_DD format)
            pattern_files.sort()
            return pattern_files
        except Exception as e:
            raise FileNotFoundError(f"Error finding image files: {e}")
    
    def _load_image(self, f):
        x = cv2.imread(f)
        if x is None:
            raise FileNotFoundError(f"[Skipped] Failed to load image file: {f}")
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = self.transform(image=x)['image']
        return x
    
    def __getitem__(self, idx):
        # Get the patient-eye group
        sample = self.eye_dfs.get_group(self.eye_ids[idx])
        
        # Get basic eye info
        patient_ids = sample['ID'].values
        lateralities = sample['LR'].values
        
        # Get survival-related eye info
        labels = sample['label'].values
        censorships = sample['censorship'].values
        event_times = sample['time_to_event'].values
        obs_times = sample['year'].values
        
        # For temporal encoding
        if self.learned_pe:
            visit_times = obs_times
        else:
            if self.tpe_mode == 'bins':
                visit_times = obs_times.astype(int)  # Use year index directly
            elif self.tpe_mode == 'months':
                visit_times = (obs_times * 12).astype(int)  # Convert year index to months
            else:
                import sys
                sys.exit(-1)
        
        # Get image directory
        patient_id = patient_ids[0]  # All should be the same
        laterality = lateralities[0]  # All should be the same
        img_dir = self._get_image_path(patient_id, laterality, self.split)
        
        # Find all image files for this patient-eye
        try:
            all_files = self._find_image_files(img_dir, patient_id, laterality)
        except Exception as e:
            print(f"[Skipped] Failed to find images for {patient_id}_{laterality}: {e}")
            # Return a random sample instead
            return self.__getitem__(np.random.randint(0, len(self.eye_ids)))
        
        # Load sequence of images
        x_seq = []
        valid_fnames = []
        
        for i, obs_time in enumerate(obs_times):
            try:
                if i < len(all_files):
                    fname = all_files[i]
                    img_path = os.path.join(img_dir, fname)
                    x = self._load_image(img_path)
                    x_seq.append(x)
                    valid_fnames.append(fname)
                else:
                    # This shouldn't happen if CSV is correct, but handle gracefully
                    print(f"[Warning] Missing image for {patient_id}_{laterality} at time {obs_time}")
                    break
            except Exception as e:
                print(f"[Skipped] Failed to load image {patient_id}_{laterality} at time {obs_time}: {e}")
                break
        
        if len(x_seq) == 0:
            # If no images loaded, return a random sample
            return self.__getitem__(np.random.randint(0, len(self.eye_ids)))
        
        # Stack images: seq_length x 3 x 224 x 224
        x_seq = np.stack(x_seq, axis=0)
        seq_length = x_seq.shape[0]
        
        # Truncate other arrays to match loaded images
        visit_times = visit_times[:seq_length]
        labels = labels[:seq_length]
        obs_times = obs_times[:seq_length]
        censorships = censorships[:seq_length]
        event_times = event_times[:seq_length]
        patient_ids = patient_ids[:seq_length]
        lateralities = lateralities[:seq_length]
        
        # Pad sequences to max_seq_length
        x_padded = np.pad(x_seq, ((0, self.max_seq_length - seq_length), (0, 0), (0, 0), (0, 0)), mode='constant')
        y_padded = np.pad(labels, (0, self.max_seq_length - seq_length), mode='constant')
        censorship_padded = np.pad(censorships, (0, self.max_seq_length - seq_length), mode='constant')
        rel_time_padded = np.pad(visit_times, (0, self.max_seq_length - seq_length), mode='constant')
        obs_time_padded = np.pad(obs_times, (0, self.max_seq_length - seq_length), mode='constant')
        
        # SIGF doesn't have AMD severity, set to None (like OHTS)
        prior_AMD_sev = np.array([-1])
        
        self.valid_sample_count += 1
        
        return {
            'x': torch.from_numpy(x_padded).float(),
            'y': torch.from_numpy(y_padded).long(),
            'censorship': torch.from_numpy(censorship_padded).float(),
            'event_time': event_times,  # Note: Not padded, handled by model logic
            'obs_time': torch.from_numpy(obs_time_padded).float(),
            'fname': valid_fnames,
            'seq_length': seq_length,
            'rel_time': torch.from_numpy(rel_time_padded).long(),
            'prior_AMD_sev': prior_AMD_sev,
            'patient_id': patient_ids,
            'laterality': lateralities
        }

### VFM-SPECIFIC SIGF DATASETS ###
class VFM_SIGF_Survival_Dataset(SIGF_Survival_Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        # Call parent constructor first
        super().__init__(data_dir, label_dir, split, augment, tpe_mode, learned_pe)
        
        # Get VFM's normalization values for Fundus images
        vfm_mean, vfm_std = get_stats('Fundus')
        
        # Override the transform to use VFM's normalization values
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 2), p=0.5),
                A.RandomResizedCrop(224, 224, scale=(0.5, 1), ratio=(0.75, 1.3333333333333333), p=1.0),
                A.Normalize(mean=vfm_mean, std=vfm_std),  # VFM's normalization for Fundus images
                albumentations.pytorch.ToTensorV2(p=1)
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=vfm_mean, std=vfm_std),  # VFM's normalization for Fundus images
                albumentations.pytorch.ToTensorV2(p=1)
            ])

class VFM_SIGF_Longitudinal_Survival_Dataset(SIGF_Longitudinal_Survival_Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        # Call parent constructor first
        super().__init__(data_dir, label_dir, split, augment, tpe_mode, learned_pe)
        
        # Get VFM's normalization values for Fundus images
        vfm_mean, vfm_std = get_stats('Fundus')
        
        # Override the transform to use VFM's normalization values
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 2), p=0.5),
                A.RandomResizedCrop(224, 224, scale=(0.5, 1), ratio=(0.75, 1.3333333333333333), p=1.0),
                A.Normalize(mean=vfm_mean, std=vfm_std),  # VFM's normalization for Fundus images
                albumentations.pytorch.ToTensorV2(p=1)
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=vfm_mean, std=vfm_std),  # VFM's normalization for Fundus images
                albumentations.pytorch.ToTensorV2(p=1)
            ])
