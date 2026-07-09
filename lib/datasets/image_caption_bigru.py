"""COCO dataset loader"""
import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
import random
import json
import nltk

import logging

logger = logging.getLogger(__name__)


def _split_tag(data_split):
    return 'train' if data_split == 'train' else 'val'


def _data_root(opt, data_path):
    root = getattr(opt, 'data_root', '')
    return root if isinstance(root, str) and root else data_path


def _pc_paths(opt, data_root, split_tag):
    grid_override = getattr(opt, f'pc_grid_{split_tag}', '')
    pos_override = getattr(opt, f'pc_pos_{split_tag}', '')
    if grid_override:
        grid_path = grid_override
    elif getattr(opt, 'pc_variant', 'base') == 'base_plus_uni3dinst':
        k = int(getattr(opt, 'uni3d_inst_k', 20))
        grid_path = osp.join(data_root, f'pt2vec_200_plus_uni3dinst_k{k}_{split_tag}.npy')
    else:
        grid_path = osp.join(data_root, f'pt2vec_200_random_{split_tag}.npy')

    if pos_override:
        pos_path = pos_override
    elif getattr(opt, 'pc_variant', 'base') == 'base_plus_uni3dinst':
        k = int(getattr(opt, 'uni3d_inst_k', 20))
        pos_path = osp.join(data_root, f'pt2vec_200_plus_uni3dinst_k{k}_pos_{split_tag}.npy')
    else:
        pos_path = osp.join(data_root, f'pt2vec_200_random_pos_{split_tag}.npy')
    return grid_path, pos_path


def _load_roma_scanrefer(data_path, data_split, opt):
    data_root = _data_root(opt, data_path)
    split_tag = _split_tag(data_split)
    grid_path, _ = _pc_paths(opt, data_root, split_tag)
    images = np.load(grid_path)
    captions = []
    with open(osp.join(data_root, f'scanrefer_{split_tag}.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            captions.append(json.loads(line)['description'])
    return captions, images, 10


class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, data_split, vocab, opt, train):
        self.vocab = vocab
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name

        if data_name == 'scanrefer':
            self.captions, self.images, self.im_div = _load_roma_scanrefer(data_path, data_split, opt)
        else:
            loc_cap = os.path.join(data_path,"precomp")
            loc_image = os.path.join(data_path,"precomp")

            # Captions
            self.captions = []
            with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
                for line in f:
                    self.captions.append(line.strip())
            # Image features
            self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % data_split))


        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if data_name == 'scanrefer':
            pass
        elif num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_name != 'scanrefer' and data_split == 'dev':
            self.length = 5000

        if hasattr(opt,"obj_drop_rate"):
            self.obj_drop_rate = opt.obj_drop_rate
        else:
            self.obj_drop_rate = 0.2

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index // self.im_div
        caption = self.captions[index]

        # Convert caption (string) to word ids (with Size Augmentation at training time).
        target = process_caption(self.vocab, caption, self.train)
        image = self.images[img_index]

        if self.train:  # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image = image[np.where(rand_list > self.obj_drop_rate)]
        image = torch.Tensor(image)
        return image, target, index, img_index

    def __len__(self):
        return self.length


def process_caption(vocab, caption, drop=False):
    if not drop:
        words = nltk.tokenize.word_tokenize(caption.lower())
        caption = list()
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in words])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return target
    else:
        # Convert caption (string) to word ids.
        tokens = ['<start>', ]
        words = nltk.tokenize.word_tokenize(caption.lower())
        tokens.extend(words)
        tokens.append('<end>')
        deleted_idx = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.20:
                prob /= 0.20
                # 50% randomly change token to mask token
                if prob < 0.5:
                    tokens[i] = vocab.word2idx['<mask>']
                # 10% randomly change token to random token
                elif prob < 0.6:
                    tokens[i] = random.randrange(len(vocab))
                # 40% randomly remove the token
                else:
                    tokens[i] = vocab(token)
                    deleted_idx.append(i)
            else:
                tokens[i] = vocab(token)
        if len(deleted_idx) != 0:
            tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]
        target = torch.Tensor(tokens)
        return target


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)
    if len(images[0].shape) == 2:  # region feature
        # Merge images
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.Tensor(img_lengths)

        # Merget captions
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return all_images, img_lengths, targets, lengths, ids
    else:  # raw input image
        # Merge images
        images = torch.stack(images, 0)
        # Merget captions
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        return images, targets, lengths, ids


def get_loader(data_path, data_name, data_split, vocab, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        drop_last = True
    else:
        drop_last = False
    if opt.precomp_enc_type in ["basic","selfattention","transformer"]:
        dset = PrecompRegionDataset(data_path, data_name, data_split, vocab, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(opt.precomp_enc_type))
    return data_loader


def get_loaders(data_path, data_name, vocab, batch_size, workers, opt):
    train_loader = get_loader(data_path, data_name, 'train', vocab, opt,
                              batch_size, True, workers, train=opt.drop)
    val_loader = get_loader(data_path, data_name, 'dev', vocab, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, vocab, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, data_name, 'train', vocab, opt,
                              batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, vocab, opt,
                             batch_size, False, workers, train=False)
    return test_loader
