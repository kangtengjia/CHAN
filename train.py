"""Training script"""
import os
import time
import json
import numpy as np
import torch
from transformers import BertTokenizer

from lib.modules import set_seeds
from lib.vocab import deserialize_vocab
from lib.datasets import image_caption_bigru,image_caption_bert
from lib.model import Model
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, compute_sims
from lib.checkpointing import build_checkpoint, restore_checkpoint
from lib.roma_config import require_bert_path, vocab_filename
from lib.roma_evaluation import DEFAULT_KS, text_to_scene_metrics
from lib.datasets.roma import canonical_dataset_name, source_paths
from lib.manifest import build_run_manifest

import logging
import tensorboard_logger as tb_logger

import lib.arguments as arguments

def main():
    # Hyper Parameters
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    set_seeds(opt.seed)

    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    logger = logging.getLogger(__name__)
    logger.info(opt)
    with open(os.path.join(opt.model_name, 'config.json'), 'w', encoding='utf-8') as handle:
        json.dump(vars(opt), handle, indent=2, default=str)

    # Load Vocabulary
    if opt.text_enc_type=="bigru":
        vocab_file = vocab_filename(opt.data_name)
        vocab = deserialize_vocab(os.path.join(opt.vocab_path, vocab_file))
        vocab.add_word('<mask>')  # add the mask, for testing cloze
        logger.info('Add <mask> token into the vocab')
        opt.vocab_size = len(vocab)
        # word embedding
        if opt.wemb_type is not None:
            opt.word2idx = vocab.word2idx
        else:
            opt.word2idx = None
        # dataloader 
        train_loader, val_loader = image_caption_bigru.get_loaders(
            opt.data_path, opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    elif opt.text_enc_type=="bert":
        opt.word2idx = None

        # Load Tokenizer and Vocabulary
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        opt.bert_path = str(require_bert_path(opt.bert_path))
        tokenizer = BertTokenizer.from_pretrained(opt.bert_path, local_files_only=True)
        vocab = tokenizer.vocab
        opt.vocab_size = len(vocab)
        train_loader, val_loader = image_caption_bert.get_loaders(
            opt.data_path, opt.data_name, tokenizer, opt.batch_size, opt.workers, opt)
    else:
        raise ValueError("Unknown text_enc_type: {}".format(opt.text_enc_type))

    model = Model(opt)
    input_manifest = {}
    if canonical_dataset_name(opt.data_name) in {'scenedepict', 'scanrefer', 'nr3d', '3dllm'}:
        manifest_paths = source_paths(opt.data_root or opt.data_path, opt.data_name, 'train')
        manifest_paths += source_paths(opt.data_root or opt.data_path, opt.data_name, 'val')
        input_manifest = build_run_manifest(dict.fromkeys(manifest_paths), repo_root=os.path.dirname(__file__))

    lr_schedules = [opt.lr_update, ]

    # optionally resume from a checkpoint
    start_epoch = 0
    best_score = 0.0
    best_metrics = {}
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            resume_state = restore_checkpoint(checkpoint, model.optimizer, weights_only=opt.weights_only)
            start_epoch = resume_state.epoch
            best_score = resume_state.best_score
            best_metrics = resume_state.best_metrics
            model.Eiters = resume_state.iterations
            logger.info("=> loaded checkpoint '{}' (epoch {}, best score {})"
                        .format(opt.resume, start_epoch, best_score))
            if opt.reset_start_epoch:
                start_epoch = 0
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))


    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        logger.info(opt.logger_name)
        logger.info(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch, lr_schedules)

        if epoch >= opt.vse_mean_warmup_epochs:
            opt.max_violation = True
            model.set_max_violation(opt.max_violation)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        metrics = validate(opt, val_loader, model)
        score = metrics['Rsum']

        # remember best R@ sum and save checkpoint
        is_best = score > best_score
        if is_best:
            best_score = score
            best_metrics = metrics
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        if is_best:logger.info("Best model saving at epoch %d"%epoch)
        checkpoint = build_checkpoint(
            epoch=epoch + 1,
            model_state=model.state_dict(),
            optimizer=model.optimizer,
            best_score=best_score,
            best_metrics=best_metrics,
            iterations=model.Eiters,
            config=vars(opt),
            scheduler_state={'milestones': lr_schedules},
            input_manifest=input_manifest,
        )
        checkpoint['opt'] = opt
        save_checkpoint(checkpoint, is_best, filename='checkpoint.pth', prefix=opt.model_name + '/')


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    if epoch<1:
        logger.info('image encoder trainable parameters: {}'.format(count_params(model.img_enc)))
        logger.info('txt encoder trainable parameters: {}'.format(count_params(model.txt_enc)))
        logger.info('similarity encoder trainable parameters: {}'.format(count_params(model.sim_enc)))

    if hasattr(train_loader.batch_sampler, 'set_epoch'):
        train_loader.batch_sampler.set_epoch(epoch)

    end = time.time()
    # opt.viz = True
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        images, img_lengths, captions, lengths, _, scene_indices = train_data
        if getattr(train_loader.dataset, 'is_roma', False) and len(scene_indices) != len(set(int(index) for index in scene_indices)):
            raise RuntimeError('scene-unique sampler emitted duplicate scenes in one batch')
        model.train_emb(images, captions, lengths, image_lengths=img_lengths)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def _encode_scene_features(model, features, batch_size):
    encoded = []
    lengths = []
    device = next(model.img_enc.parameters()).device
    for start in range(0, len(features), int(batch_size)):
        batch = torch.as_tensor(features[start:start + int(batch_size)], dtype=torch.float32, device=device)
        batch_lengths = torch.full((len(batch),), batch.size(1), dtype=torch.float32, device=device)
        output = model.img_enc(batch, batch_lengths)
        encoded.append(output.detach().cpu().numpy())
        lengths.extend([batch.size(1)] * len(batch))
    return np.concatenate(encoded, axis=0), lengths


def validate(opt, val_loader, model):
    logger = logging.getLogger(__name__)
    model.val_start()
    with torch.no_grad():
        # compute the encoding for all the validation images and captions
        _, cap_embs, _, cap_lens, caption_scene_indices = encode_data(
            model, val_loader, opt.log_step, logging.info, return_scene_indices=True
        )
        img_embs, img_lens = _encode_scene_features(model, val_loader.dataset.images, opt.batch_size)

    start = time.time()
    # sims = compute_sim(img_embs, cap_embs)
    with torch.no_grad():
        sims = compute_sims(img_embs, cap_embs, img_lens, cap_lens, model)
    end = time.time()
    logger.info("calculate similarity time:".format(end - start))

    metrics = text_to_scene_metrics(sims.T, caption_scene_indices, ks=DEFAULT_KS)
    logger.info('Text to scene: R@1 %.2f R@5 %.2f R@10 %.2f R@30 %.2f MedR %.1f MeanR %.1f Rsum %.2f',
                metrics['R@1'], metrics['R@5'], metrics['R@10'], metrics['R@30'],
                metrics['MedR'], metrics['MeanR'], metrics['Rsum'])

    # record metrics in tensorboard
    for key in ['R@1', 'R@5', 'R@10', 'R@30', 'MedR', 'MeanR', 'Rsum']:
        tb_logger.log_value(key.replace('@', ''), metrics[key], step=model.Eiters)
    return metrics

def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 15

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch, lr_schedules):
    logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    main()
