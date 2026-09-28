from torch.utils.data import Dataset, Sampler
from pathlib import Path
from utils.partitioning import *
from utils.misc_utils import *
from utils.ktgaussian import ktgaussian_mask
import numpy as np
import os
import random
import csv
import h5py
import re


def _nx_from_params(subject_dir) -> int:
    """Read the kx (Nx) dimension from params.csv — avoids opening the large mat file."""
    with open(Path(subject_dir) / 'params.csv') as f:
        row = next(csv.DictReader(f))
    return int(row['matrix_size'].split(';')[5])  # FE (readout) = recon-slice (kx) axis


def _ifft_readout(x):
    """Centered orthonormal inverse FFT along the readout axis (last axis = FE)."""
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=-1), axis=-1, norm="ortho"), axes=-1)


def mat_to_npy_kdata_val(file):
	with h5py.File(file, "r") as f:
		arr = np.array(f["kdata_ktGaussian"])
	return arr

def mat_to_npy_kdata(file, venc=None, bins=None):
	with h5py.File(file, "r") as f:
		dset = f["kdata_full"]
		vsl = slice(None) if venc is None else slice(int(venc), int(venc) + 1)
		if bins is None:
			arr = np.array(dset[vsl])
		else:
			b = np.asarray(bins)
			uniq, inv = np.unique(b, return_inverse=True)   # h5py needs increasing indices
			arr = dset[vsl, list(uniq)][:, inv]             # gather back to requested bin order
	return arr


def mat_to_npy_coilmap(file):
	with h5py.File(file, "r") as f:
		arr = np.array(f["coilmap"])
	return arr
	
def mat_to_npy_segmask(file):
	with h5py.File(file, "r") as f:
		arr = np.array(f["segmask"])
	return arr


class SubjectGroupedSampler(Sampler):
	"""Yield dataset indices grouped by subject so a worker reads each subject's
	large kdata_full.mat once per epoch (a single cache fill in _read_full_volume)
	instead of re-reading it for every readout slice. Subject order is reshuffled
	every epoch, and so are the slices within each subject.

	DDP-aware: subjects are sharded disjointly across ranks (so each GPU still reads
	each of its files once), and every rank is padded to the same length so DDP can't
	deadlock at the end of an epoch. With world_size=1 it behaves exactly as before."""
	def __init__(self, dataset, shuffle=True, seed=None, num_replicas=None, rank=None,
	             scanner_reps=None):
		self.groups = dataset.subject_groups()      # list[list[int]] over self.filename
		if scanner_reps and hasattr(dataset, 'subject_scanners'):
			scanners = dataset.subject_scanners()
			expanded = []
			for g, s in zip(self.groups, scanners):
				expanded.extend([g] * max(1, int(scanner_reps.get(s, 1))))
			self.groups = expanded
		self.shuffle = shuffle
		self.seed = 0 if seed is None else int(seed)
		self.epoch = 0
		# resolve DDP rank / world size (default: single process)
		if num_replicas is None or rank is None:
			if torch.distributed.is_available() and torch.distributed.is_initialized():
				num_replicas = torch.distributed.get_world_size()
				rank = torch.distributed.get_rank()
			elif int(os.environ.get('WORLD_SIZE', 1)) > 1:
				num_replicas = int(os.environ['WORLD_SIZE'])
				rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
			else:
				num_replicas, rank = 1, 0
		self.num_replicas, self.rank = num_replicas, rank
		self._my_groups = self.groups[self.rank::self.num_replicas]
		per_rank = [sum(len(g) for g in self.groups[r::self.num_replicas]) for r in range(self.num_replicas)]
		self._len = max(per_rank) if per_rank else 0

	def set_epoch(self, epoch):
		self.epoch = int(epoch)

	def __len__(self):
		return self._len

	def __iter__(self):
		groups = [list(g) for g in self._my_groups]
		if self.shuffle:
			rng = random.Random(self.seed + self.epoch)
			rng.shuffle(groups)                     # shuffle subject order
			for g in groups:
				rng.shuffle(g)                      # shuffle slices within a subject
			self.epoch += 1                         # advance so the next epoch reshuffles
		idxs = [i for g in groups for i in g]
		if idxs and len(idxs) < self._len:          # pad this rank to the global max (DDP-safe)
			reps = -(-self._len // len(idxs))   # ceil-divide so even tiny shards reach _len
			idxs = (idxs * reps)[:self._len]
		return iter(idxs)


class OwnDataset(Dataset):
	def __init__(self, **kwargs):
		options = dict(kwargs)
		self.options = options.copy()
		self.filename = []
		self.valfilename = []
		self._vol_cache_key = None
		self._vol_cache = None
		self.D_size = options['D_size']
		self.T_size = options['T_size']
		self.input = options['input']
		self.val_dir = options['val_dir']
		self.loss = options['loss']
		self.augment = options.get('augment', False)   # train-time source augmentation
		self.aug_noise = float(options.get('aug_noise', 0.0))   # input-only k-space noise level
		self.aug_coil  = float(options.get('aug_coil', 0.0))    # coil-map (CSM) perturbation level
		self.aug_phase_ramp = float(options.get('aug_phase_ramp', 0.0))   # #12 B0/eddy phase-ramp aug (cycles across FE)
		self.robust_norm = bool(options.get('robust_norm', False))   # winsorized-RMS k-space scale
		self._mask_pool_size = int(options.get('mask_pool', 0) or 0)  # see _apply_train_mask
		self._mask_pool = {}
		self.is_brain = 'brain' in (options.get('root_dir') or '')
		# _val_use_kdata_full: True when internal val uses kdata_full + simulate_undersampling
		self._val_use_kdata_full = options.get('explicit_val_dirs') is not None
		self._target_acc = options.get('target_acc', None)
		patient_dir = []
		file_all = []

		if self.options['mode'] == 'train':
			if options.get('explicit_train_dirs') is not None:
				patient_dir = [Path(d) for d in options['explicit_train_dirs']]
			else:
				for file in sorted(Path(self.options['root_dir']).rglob("*.csv")):
					if file.is_file():
						patient_dir.append(file.parent)

			# 'full' subjects: kdata_full source, simulate undersampling (Aorta train).
			for patient in patient_dir:
				Nx = _nx_from_params(patient)
				for i in range(Nx - self.D_size + 1):
					self.filename.append([patient, i, 'full'])

			for d in (options.get('train_undersampled_dirs') or []):
				patient = Path(d)
				Nx = _nx_from_params(patient)
				for i in range(Nx - self.D_size + 1):
					self.filename.append([patient, i, 'us'])
		else:
			if self._val_use_kdata_full:
				# Internal val from training subjects — load kdata_full at __getitem__ time
				for patient in [Path(d) for d in options['explicit_val_dirs']]:
					file_path = patient / self.input
					Nx = _nx_from_params(patient)
					for i in range(Nx - self.D_size + 1):
						self.valfilename.append([patient, i])
			else:
				# External val from val_dir — load kdata_ktGaussian
				pattern = "kdata_ktGaussian[0-9]+.mat"
				for file in sorted(Path(self.options['val_dir']).rglob("*")):
					if file.is_file():
						if len(re.findall(pattern, str(file))) == 1:
							file_all.append(file)
				for singfile in file_all:
					Nx = _nx_from_params(singfile.parent)
					for i in range(Nx - self.D_size + 1):
						self.valfilename.append([singfile.parent, i])

	def _kspace_norm(self, f):
		"""Per-sample k-space scale used to normalize the input (and to undo on recon).
		Default: RMS over the sampled samples. Robust (--robust_norm): the same RMS but
		with the top 0.5% magnitudes winsorized, so a handful of vendor-specific hot
		samples / bright central lines can't set the scale -> steadier cross-center
		statistics. Scale stays comparable to the plain-RMS value, so loss thresholds
		and the stored 'norm' rescaling carry over unchanged."""
		nz = f[f != 0]
		if nz.size == 0:
			return 1.0
		if self.robust_norm:
			mag = np.abs(nz)
			mag = np.minimum(mag, np.percentile(mag, 99.5))
			return float(np.sqrt(np.mean(mag ** 2)))
		return float(np.linalg.norm(f) / np.linalg.norm(f != 0))

	def _perturb_coil(self, c):
		"""CSM-error augmentation (#3): multiplicative complex perturbation of the coil
		maps, then RSS-renormalized. Per-sample sigma ~ U(0, aug_coil). Trains the net to
		reconstruct despite imperfect/varying coil sensitivities (cross-scanner robustness).
		c: (Nc, D, ky, kz) -> perturbed, unit-RSS c (same shape)."""
		sigma = np.random.uniform(0, self.aug_coil)
		pert = (1.0 + sigma * (np.random.randn(*c.shape) + 1j * np.random.randn(*c.shape)) / np.sqrt(2)).astype(np.complex64)
		c = (c * pert).astype(np.complex64)
		rss = np.sqrt(np.sum(np.abs(c) ** 2, axis=0, keepdims=True)) + 1e-8
		return (c / rss).astype(np.complex64)

	def _apply_train_mask(self, f):
		"""Simulate the challenge's k-t Gaussian undersampling at a random R.

		ktgaussian_mask runs a rejection-sampling Python loop per ACCEPTED k-space point,
		once per time frame, so a fresh mask on every __getitem__ is expensive -- and jssl
		pays it TWICE per step (both streams draw one). --mask_pool N keeps a small
		per-worker pool per (ky,kz,T,R) and reuses a random member once the pool is full,
		amortizing generation to N calls per (shape,R). The SSDU p1/p2 split and every
		augmentation are still redrawn per sample, so the stochasticity that matters is
		unaffected; only the set of distinct acquisition masks is finite. --mask_pool 0
		(default) restores fresh-every-sample."""
		V, C, T, D, ky, kz = f.shape
		usrate = self._target_acc if self._target_acc is not None else random.choice([10, 20, 30, 40, 50])
		mask_tkk = None
		key = (int(ky), int(kz), int(T), int(usrate))
		if self._mask_pool_size > 0:
			pool = self._mask_pool.setdefault(key, [])
			if len(pool) >= self._mask_pool_size:
				mask_tkk = pool[np.random.randint(len(pool))]
		if mask_tkk is None:
			mask_tkk = ktgaussian_mask(ky, kz, T, int(usrate))   # (T, ky, kz) kt-Gaussian (matches challenge)
			if self._mask_pool_size > 0:
				self._mask_pool.setdefault(key, []).append(mask_tkk)
		return np.broadcast_to(mask_tkk[np.newaxis], (V, T, ky, kz)).copy(), int(usrate)

	def _read_full_volume(self, sd, internal, itemwant=None):
		"""Read+iFFT+transpose a subject's full (all-readout) volume once and cache it, so the
		many stride-1 slabs of one subject don't each re-read the whole file (1 entry, per worker)."""
		key = (str(sd), bool(internal), itemwant)
		if self._vol_cache_key == key:
			return self._vol_cache
		if internal:
			f = mat_to_npy_kdata(sd / self.input)
		else:
			f = mat_to_npy_kdata_val(Path(str(sd) + "/" + itemwant))
		f = f['real'] + f['imag'] * 1j
		f = f.astype(np.complex64)
		f = _ifft_readout(f)
		f = np.transpose(f, (0, 2, 1, 5, 4, 3))            # (Nv, Nc, T, FE, PE, SPE)
		c = mat_to_npy_coilmap(sd / 'coilmap.mat')
		c = c['real'] + c['imag'] * 1j
		c = c.astype(np.complex64)
		c = np.transpose(c, (0, 3, 2, 1))                  # (Nc, FE, PE, SPE)
		c = c / (np.sqrt(np.sum(np.abs(c) ** 2, axis=0, keepdims=True)) + 1e-8)  # unit-RSS coil norm
		s = mat_to_npy_segmask(sd / 'segmask.mat')         # (SPE, PE, FE)
		self._vol_cache_key = key
		self._vol_cache = (f, c, s)
		return self._vol_cache

	def __len__(self):
		if self.options['mode'] == 'train':
			return len(self.filename)
		return len(self.valfilename)

	def get_order(self):
		if self.options['mode'] == 'train':
			return self.filename
		return self.valfilename

	def subject_groups(self):
		"""Index-lists (over self.filename), one per subject, for SubjectGroupedSampler.
		Keeps a worker on a single subject so its k-space volume is read once per epoch."""
		groups = {}
		for i, entry in enumerate(self.filename):
			patient = entry[0]   # entry may be [patient, slice] or [patient, slice, src]
			groups.setdefault(str(patient), []).append(i)
		return list(groups.values())

	def subject_scanners(self):
		"""Scanner tag (immediate parent dir of the patient dir, e.g. 'UIH_30T_uMR890') per
		subject, in the SAME order as subject_groups() -- lets SubjectGroupedSampler oversample
		under-represented scanners (--scanner_reps). Keyed by patient dir exactly like
		subject_groups so the two lists stay index-aligned."""
		scan = {}
		for entry in self.filename:
			patient = entry[0]
			scan.setdefault(str(patient), Path(patient).parent.name)
		return list(scan.values())

	def __getitem__(self, idx):

		if self.options['mode'] == 'train':
			
			patient = self.filename[idx][0]
			fe = self.filename[idx][1]
			src = self.filename[idx][2] if len(self.filename[idx]) > 2 else 'full'
			us_R = None
			if src == 'us':
				_us = sorted(patient.glob("kdata_ktGaussian[0-9]*.mat"))
				if not _us:
					raise FileNotFoundError(f"[dataloader] no kdata_ktGaussian*.mat in {patient}")
				_m = re.search(r'ktGaussian(\d+)', _us[0].name)
				us_R = int(_m.group(1)) if _m else 0
				f_full, c_full, s_full = self._read_full_volume(patient, internal=False, itemwant=_us[0].name)
			else:
				f_full, c_full, s_full = self._read_full_volume(patient, internal=True)
			# f_full:(Nv,Nc,T,FE,ky,kz)  c_full:(Nc,FE,ky,kz)  s_full:(SPE,PE,FE)  (c already RSS-normalized)
			Nv, total_bins = f_full.shape[0], f_full.shape[2]
			_allvenc = (self.options['network'] == "FlowVN"
			            and bool(self.options.get('flowvn_allvenc', False))
			            and self.loss == 'supervised')
			seg_read = None if _allvenc else (np.random.randint(Nv) if self.options['network'] == "FlowVN" else None)
			if self.T_size == -1:
				cardiac_bins = list(range(total_bins))
			else:
				first_bin = random.randint(-self.T_size+1, total_bins-self.T_size)  # any cardiac bin can be the center (wraps)
				cardiac_bins = list(np.mod(np.arange(first_bin, first_bin+self.T_size), total_bins))
			vsl = slice(None) if seg_read is None else slice(seg_read, seg_read+1)
			f = f_full[vsl][:, :, cardiac_bins, fe:fe+self.D_size, :, :].copy()  # copy: never mutate the cache
			c = c_full[:, fe:fe+self.D_size, :, :]
			s = s_full[:, :, fe:fe+self.D_size]   # (SPE, PE, D) slab matching the recon

			if self.augment:
				if np.random.rand() < 0.5:                       # readout (FE) flip; ky/kz are k-space, left alone
					f = f[:, :, :, ::-1, :, :].copy()
					c = c[:, ::-1, :, :].copy()
					s = s[:, :, ::-1].copy()
				phase = np.complex64(np.exp(1j * np.random.uniform(0, 2 * np.pi)))  # global phase: magnitude- & velocity-invariant
				f = f * phase

				if self.aug_phase_ramp > 0:
					FE = f.shape[3]
					alpha = np.random.uniform(-self.aug_phase_ramp, self.aug_phase_ramp)   # cycles across the FE extent
					ramp = np.exp(1j * 2 * np.pi * alpha * (np.arange(FE, dtype=np.float32) / max(FE, 1))).astype(np.complex64)
					f = f * ramp[None, None, None, :, None, None]

			mask = (abs(f[:,0,:,0,:,:]) != 0).astype(np.float32)
			
		im_gt = None  # holds the fully-sampled GT image when full k-space is available
		if self.options['mode'] in ('val', 'test'):
			usrate = 0  # will be updated in each val branch
			if self._val_use_kdata_full and self.options['mode'] == 'val':
				# Internal val: kdata_full + deterministic simulate_undersampling
				usrate = int(self._target_acc) if self._target_acc else int(np.random.RandomState(idx).choice([10, 20, 30, 40, 50]))
				f_full, c_full, s_full = self._read_full_volume(self.valfilename[idx][0], internal=True)
				ss = self.valfilename[idx][1]
				f = f_full[:, :, :, ss:ss+self.D_size, :, :].copy()
				c = c_full[:, ss:ss+self.D_size, :, :]
				s = s_full[:, :, ss:ss+self.D_size]
				im_gt = mriAdjointOp(f, c[np.newaxis, :, np.newaxis, :, :, :], np.ones_like(f[:, 0, :, 0, :, :], dtype=np.float32)[:, np.newaxis, :, np.newaxis]).astype(np.complex64)
				np.random.seed(idx)   # deterministic kt-Gaussian -> reproducible val
				mask_tkk = ktgaussian_mask(f.shape[-2], f.shape[-1], f.shape[2], usrate)
				sim_mask = np.broadcast_to(mask_tkk[np.newaxis], (f.shape[0], f.shape[2], f.shape[-2], f.shape[-1])).copy()
				f = f * sim_mask[:, np.newaxis, :, np.newaxis]
				mask = (abs(f[:, 0, :, 0, :, :]) != 0).astype(np.float32)
			else:
				# External val: kdata_ktGaussian from val_dir
				np.random.seed(0)
				pattern = "kdata_ktGaussian[0-9]+.mat"
				items_all = [re.findall(pattern, str(k)) for k in os.listdir(self.valfilename[idx][0])]
				for i in items_all:
					if len(i) != 0:
						itemwant = i[0]
				_m = re.search(r'ktGaussian(\d+)', itemwant)
				usrate = int(_m.group(1)) if _m else 0
				f_full, c_full, s_full = self._read_full_volume(self.valfilename[idx][0], internal=False, itemwant=itemwant)
				ss = self.valfilename[idx][1]
				f = f_full[:, :, :, ss:ss+self.D_size, :, :].copy()
				c = c_full[:, ss:ss+self.D_size, :, :]
				s = s_full[:, :, ss:ss+self.D_size]
				mask = (abs(f[:, 0, :, 0, :, :]) != 0).astype(np.float32)
				if np.sum(mask) == 0:
					print(f"[dataloader] empty mask at val idx={idx} "
					      f"({self.valfilename[idx][0].name}, D={self.valfilename[idx][1]}), skipping")
					return self.__getitem__((idx + 1) % len(self.valfilename))

		_allvenc = (self.options['network'] == "FlowVN"
		            and bool(self.options.get('flowvn_allvenc', False))
		            and self.loss == 'supervised' and self.options['mode'] == 'train')
		seg_idx = 0 if _allvenc else (np.random.randint(f.shape[0]) if (self.options['network'] == "FlowVN" and self.options['mode'] == 'train') else 0)
		if self.options['network'] == "FlowVN" and self.options['mode'] == 'train' and not _allvenc:  # segment-wise only in train (val/test keep all vencs)
			f = f[[seg_idx]]
			mask = mask[[seg_idx]]

		_ssdu_split = (self.loss == 'ssdu') and not (self.options['mode'] == 'val' and im_gt is not None)
		if _ssdu_split:   # k-space split: train + external val (no fully-sampled GT)
			if self.options['mode'] == 'train':
				if src == 'us':
					mask = (abs(f[:, 0, :, 0, :, :]) != 0).astype(np.float32)
					train_usrate = int(us_R)   # nominal R from the ktGaussian filename
				else:
					# Simulate undersampling to match validation k-space density (~2-10%)
					sim_mask, train_usrate = self._apply_train_mask(f)
					f = f * sim_mask[:, np.newaxis, :, np.newaxis]
					mask = (abs(f[:, 0, :, 0, :, :]) != 0).astype(np.float32)
				mask_p1, mask_p2 = uniform_disjoint_selection(mask, rho=self.options.get('ssdu_rho', 0.2), r2=9, seed=None, venc_coherence=True)
			elif self.options['mode'] == 'val':
				mask_p1, mask_p2 = uniform_disjoint_selection(mask, rho=self.options.get('ssdu_rho', 0.2), r2=9, seed=(None if self.options.get('ssl_random_split') else 444), venc_coherence=True)
			else:
				mask_p1, mask_p2 = mask, np.zeros_like(mask)

			# partition data
			kdata_p1 = f * mask_p1[:, np.newaxis, :, np.newaxis]
			kdata_p2 = f * mask_p2[:, np.newaxis, :, np.newaxis]

			if self.options['mode'] == 'train' and self.aug_noise > 0:
				nz = kdata_p1[kdata_p1 != 0]
				sig = float(np.sqrt(np.mean(np.abs(nz) ** 2))) if nz.size else 0.0
				sigma = np.float32(np.random.uniform(0, self.aug_noise) * sig / np.sqrt(2))
				noise = (np.random.randn(*kdata_p1.shape) + 1j * np.random.randn(*kdata_p1.shape)).astype(np.complex64) * sigma
				kdata_p1 = kdata_p1 + noise * mask_p1[:, np.newaxis, :, np.newaxis]

			norm = self._kspace_norm(f)
			kdata_p1 /= norm
			kdata_p2 /= norm

			kdata_p2 = torch.from_numpy(kdata_p2)
			kdata_p1 = torch.from_numpy(kdata_p1)
			if self.options['mode'] == 'train' and self.aug_coil > 0:
				c = self._perturb_coil(c)        # CSM-error augmentation (#3)
			c = torch.from_numpy(c.copy())  # c may be a cache view; copy protects the cache

			if self.options['mode'] == 'train':
				usrate_true_val = float(1.0 / max(float(np.mean(mask)), 1e-6))
				return {'kdata_p2': kdata_p2, 'kdata_p1': kdata_p1, 'coil_sens': c,
				        'norm': norm, "segmentation": s.astype(bool), "subj": str(self.filename[idx][0].stem).split('/')[-1],
				        'usrate': train_usrate, 'usrate_true': torch.tensor([usrate_true_val], dtype=torch.float32)}
			else:  # val or test
				usrate_true_val = float(1.0 / max(float(np.mean(mask)), 1e-6))
				return {'kdata_p2': kdata_p2, 'kdata_p1': kdata_p1, 'coil_sens': c,
				        'norm': norm, "segmentation": s.astype(bool), "subj": str(self.valfilename[idx][0].stem).split('/')[-1],
				        'slice_start': self.valfilename[idx][1], 'case_dir': str(self.valfilename[idx][0]),
				        'usrate': usrate, 'usrate_true': torch.tensor([usrate_true_val], dtype=torch.float32)}

		elif self.loss == 'supervised' or (self.loss == 'ssdu' and self.options['mode'] == 'val' and im_gt is not None):   # supervised, or ssdu internal val -> leaderboard-style GT
			v_slice = slice(None) if _allvenc else (slice(seg_idx, seg_idx + 1) if (self.options['network'] == "FlowVN" and self.options['mode'] == 'train') else slice(None))

			if im_gt is not None:
				# internal val: true GT captured before the val undersampling
				im = im_gt[v_slice][np.newaxis].astype(np.complex64)
			else:
				im = mriAdjointOp(f, c[np.newaxis, :, np.newaxis, :, :, :], np.ones_like(f[:, 0, :, 0, :, :], dtype=np.float32)[:, np.newaxis, :, np.newaxis]).astype(np.complex64)
				im = im[np.newaxis]

			if self.options['mode'] == 'train':
				# Simulate undersampling to match validation k-space density (~2-10%)
				sim_mask, train_usrate = self._apply_train_mask(f)
				f = f * sim_mask[:, np.newaxis, :, np.newaxis]
				mask = (abs(f[:, 0, :, 0, :, :]) != 0).astype(np.float32)

				if self.aug_noise > 0:
					nz = f[f != 0]
					sig = float(np.sqrt(np.mean(np.abs(nz) ** 2))) if nz.size else 0.0
					sigma = np.float32(np.random.uniform(0, self.aug_noise) * sig / np.sqrt(2))
					noise = (np.random.randn(*f.shape) + 1j * np.random.randn(*f.shape)).astype(np.complex64) * sigma
					f = f + noise * sim_mask[:, np.newaxis, :, np.newaxis]   # only on sampled locations; unsampled stay 0


			if self.options['mode'] == 'train':
				subj_key = str(self.filename[idx][0].stem).split('/')[-1]
			else:
				subj_key = str(self.valfilename[idx][0].stem).split('/')[-1]

			# Normalize data
			norm = self._kspace_norm(f)
			im = im / norm
			f /= norm

			im = torch.from_numpy(im)
			f = torch.from_numpy(f)
			if self.options['mode'] == 'train' and self.aug_coil > 0:
				c = self._perturb_coil(c)        # CSM-error aug (#3); GT 'im' above used the true c
			c = torch.from_numpy(c.copy())  # c may be a cache view; copy protects the cache

			if self.options['mode'] == 'train':
				usrate_true_val = float(1.0 / max(float(np.mean(mask)), 1e-6))
				return {'gt': im, 'kdata_p1': f, 'coil_sens': c,
				        'norm': norm, "segmentation": s.astype(bool), "subj": subj_key, "seg_idx": int(seg_idx),
								'slice_start': self.filename[idx][1], 'case_dir': str(self.filename[idx][0]),
				        'usrate': train_usrate, 'usrate_true': torch.tensor([usrate_true_val], dtype=torch.float32)}
			else:  # val or test
				usrate_true_val = float(1.0 / max(float(np.mean(mask)), 1e-6))
				return {'gt': im, 'kdata_p1': f, 'coil_sens': c,
				        'norm': norm, "segmentation": s.astype(bool), "subj": subj_key, "seg_idx": int(seg_idx),
				        'slice_start': self.valfilename[idx][1], 'case_dir': str(self.valfilename[idx][0]),
				        'usrate': usrate, 'usrate_true': torch.tensor([usrate_true_val], dtype=torch.float32)}

		else:
			raise ValueError("loss must be 'ssdu' or 'supervised' (jssl combines both)")


class JSSLDataset(Dataset):
	"""Joint Supervised + Self-Supervised (arXiv:2311.15856): each item pairs a
	fully-sampled (supervised) sample with a random undersampled-target (SSDU) sample.
	__len__ = max(...) so the smaller stream is effectively oversampled."""
	def __init__(self, sup_ds, ssl_ds):
		self.sup_ds = sup_ds
		self.ssl_ds = ssl_ds
		self._ssl_groups = ssl_ds.subject_groups()
		self._ssl_g = None
		self._ssl_left = 0
	def __len__(self):
		return max(len(self.sup_ds), len(self.ssl_ds))
	def subject_groups(self):
		"""Index groups over THIS dataset, so SubjectGroupedSampler can drive it and the
		supervised stream's volume cache hits too. __getitem__ maps i -> sup_ds[i % len(sup_ds)],
		and sup_ds's groups are already index-lists over exactly that range."""
		return self.sup_ds.subject_groups()
	def subject_scanners(self):
		"""Scanner tags aligned with subject_groups(); delegates to the supervised stream that
		drives the sampler (so --scanner_reps rebalances the supervised subjects)."""
		return self.sup_ds.subject_scanners()
	def __getitem__(self, i):
		sup = self.sup_ds[i % len(self.sup_ds)]
		if self._ssl_left <= 0:
			self._ssl_g = self._ssl_groups[random.randrange(len(self._ssl_groups))]
			self._ssl_left = len(self._ssl_g)
		self._ssl_left -= 1
		ssl = self.ssl_ds[self._ssl_g[random.randrange(len(self._ssl_g))]]
		return {"sup": sup, "ssl": ssl}
