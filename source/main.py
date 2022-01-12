import argparse
import itertools
import os.path
import time
import sys

import torch
import torch.optim.lr_scheduler

import numpy as np
import math
import evaluate
import trees
import vocabulary
import makehp
import Lparser
import utils
import json

tokens = Lparser


def torch_load(load_path):
    if Lparser.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def make_hparams():
    return makehp.HParams(
        max_len_train=0,  # no length limit
        max_len_dev=0,  # no length limit
        sentence_max_len=300,
        learning_rate=0.0008,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0.0,  # no clipping
        step_decay=True,  # note that disabling step decay is not implemented
        step_decay_factor=0.5,
        step_decay_patience=5,
        partitioned=True,
        use_cat=False,
        const_lada=0.5,
        num_layers=12,
        d_model=1024,
        num_heads=8,
        d_kv=64,
        d_ff=2048,
        d_label_hidden=250,
        d_biaffine=1024,
        attention_dropout=0.33,
        embedding_dropout=0.33,
        relu_dropout=0.33,
        residual_dropout=0.33,
        use_tags=False,
        use_words=False,
        use_elmo=False,
        use_bert=False,
        use_xlnet=False,
        use_bert_only=False,
        use_chars_lstm=False,
        dataset="ptb",
        model_name="joint",
        # ['glove','sskip','random']
        embedding_type="glove",
        embedding_path="./data/glove/glove.gz",
        punctuation="." "``" "''" ":" ",",
        d_char_emb=64,
        tag_emb_dropout=0.33,
        word_emb_dropout=0.33,
        morpho_emb_dropout=0.33,
        timing_dropout=0.0,
        char_lstm_input_dropout=0.33,
        elmo_dropout=0.5,
        bert_model_path="./data/bert/large-uncased",
        bert_do_lower_case=True,
        bert_transliterate="",
        xlnet_model="xlnet-large-cased",
        xlnet_do_lower_case=False,
        pad_left=False,
    )


def run_train(args, hparams):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    seed_from_numpy = np.random.randint(2147483648)
    print("Manual seed for pytorch:", seed_from_numpy)
    torch.manual_seed(seed_from_numpy)

    hparams.set_from_args(args)
    print("Hyperparameters:")
    hparams.print()

    train_path = args.train_ptb_path
    dev_path = args.dev_ptb_path

    if hparams.dataset == "ctb":
        train_path = args.train_ctb_path
        dev_path = args.dev_ctb_path

    print("Loading training trees from {}...".format(train_path))
    train_treebank = trees.load_trees(train_path)
    if hparams.max_len_train > 0:
        train_treebank = [
            tree
            for tree in train_treebank
            if len(list(tree.leaves())) <= hparams.max_len_train
        ]
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(dev_path))
    dev_treebank = trees.load_trees(dev_path)
    if hparams.max_len_dev > 0:
        dev_treebank = [
            tree
            for tree in dev_treebank
            if len(list(tree.leaves())) <= hparams.max_len_dev
        ]
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")
    train_parse = [tree.convert() for tree in train_treebank]
    dev_parse = [tree.convert() for tree in dev_treebank]

    print("Constructing vocabularies...")

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(Lparser.START)
    tag_vocab.index(Lparser.STOP)
    tag_vocab.index(Lparser.TAG_UNK)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(Lparser.START)
    word_vocab.index(Lparser.STOP)
    word_vocab.index(Lparser.UNK)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(())

    char_set = set()

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                label_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)
                char_set |= set(node.word)

    char_vocab = vocabulary.Vocabulary()

    # If codepoints are small (e.g. Latin alphabet), index by codepoint directly
    highest_codepoint = max(ord(char) for char in char_set)
    if highest_codepoint < 512:
        if highest_codepoint < 256:
            highest_codepoint = 256
        else:
            highest_codepoint = 512

        # This also takes care of constants like tokens.CHAR_PAD
        for codepoint in range(highest_codepoint):
            char_index = char_vocab.index(chr(codepoint))
            assert char_index == codepoint
    else:
        char_vocab.index(tokens.CHAR_UNK)
        char_vocab.index(tokens.CHAR_START_SENTENCE)
        char_vocab.index(tokens.CHAR_START_WORD)
        char_vocab.index(tokens.CHAR_STOP_WORD)
        char_vocab.index(tokens.CHAR_STOP_SENTENCE)
        for char in sorted(char_set):
            char_vocab.index(char)

    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()
    char_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {tokens.START, tokens.STOP, tokens.UNK}
        print(
            "{} ({:,}): {}".format(
                name,
                vocab.size,
                sorted(value for value in vocab.values if value in special)
                + sorted(value for value in vocab.values if value not in special),
            )
        )

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)
        print_vocabulary("Char", char_vocab)

    print("Initializing model...")

    load_path = None
    if load_path is not None:
        print("Loading parameters from {}".format(load_path))
        info = torch_load(load_path)
        parser = Lparser.ChartParser.from_spec(info["spec"], info["state_dict"])
    else:
        parser = Lparser.ChartParser(
            tag_vocab,
            word_vocab,
            label_vocab,
            char_vocab,
            hparams,
        )

    print("Initializing optimizer...")
    trainable_parameters = [
        param for param in parser.parameters() if param.requires_grad
    ]
    trainer = torch.optim.Adam(
        trainable_parameters, lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    if load_path is not None:
        trainer.load_state_dict(info["trainer"])

    def set_lr(new_lr):
        for param_group in trainer.param_groups:
            param_group["lr"] = new_lr

    assert hparams.step_decay, "Only step_decay schedule is supported"

    warmup_coeff = hparams.learning_rate / hparams.learning_rate_warmup_steps
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer,
        "max",
        factor=hparams.step_decay_factor,
        patience=hparams.step_decay_patience,
        verbose=True,
    )

    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= hparams.learning_rate_warmup_steps:
            set_lr(iteration * warmup_coeff)

    clippable_parameters = trainable_parameters
    grad_clip_threshold = (
        np.inf if hparams.clip_grad_norm == 0 else hparams.clip_grad_norm
    )

    print("Training...")
    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_fscore = -np.inf
    best_model_path = None
    model_name = hparams.model_name
    best_dev_processed = 0

    print("This is ", model_name)
    start_time = time.time()

    def check_dev(epoch_num):
        nonlocal best_dev_fscore
        nonlocal best_model_path
        nonlocal best_dev_processed

        dev_start_time = time.time()

        parser.eval()

        dev_predicted = []

        for dev_start_index in range(0, len(dev_treebank), args.eval_batch_size):
            subbatch_trees = dev_treebank[
                dev_start_index : dev_start_index + args.eval_batch_size
            ]
            subbatch_sentences = [
                [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                for tree in subbatch_trees
            ]

            (
                predicted,
                _,
            ) = parser.parse_batch(subbatch_sentences)
            del _

            dev_predicted.extend([p.convert() for p in predicted])
        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted)

        print(
            "\n"
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore, format_elapsed(dev_start_time), format_elapsed(start_time)
            )
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = best_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_model_path = "{}_best_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore
            )
            best_dev_processed = total_processed
            print("Saving new best model to {}...".format(best_model_path))
            torch.save(
                {
                    "spec": parser.spec,
                    "state_dict": parser.state_dict(),
                    "trainer": trainer.state_dict(),
                },
                best_model_path + ".pt",
            )

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            trainer.zero_grad()
            schedule_lr(total_processed // args.batch_size)

            parser.train()

            batch_loss_value = 0.0
            batch_trees = train_parse[start_index : start_index + args.batch_size]

            batch_sentences = [
                [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                for tree in batch_trees
            ]
            for subbatch_sentences, subbatch_trees in parser.split_batch(
                batch_sentences, batch_trees, args.subbatch_max_tokens
            ):
                _, loss = parser.parse_batch(subbatch_sentences, subbatch_trees)

                loss = loss / len(batch_trees)
                loss_value = float(loss.data.cpu().numpy())
                batch_loss_value += loss_value
                if loss_value > 0:
                    loss.backward()
                del loss
                total_processed += len(subbatch_trees)
                current_processed += len(subbatch_trees)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                clippable_parameters, grad_clip_threshold
            )

            trainer.step()

            print(
                "\r"
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "grad-norm {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_parse) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    grad_norm,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                ),
                end="",
            )
            sys.stdout.flush()

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev(epoch)

        # adjust learning rate at the end of an epoch
        if hparams.step_decay:
            if (
                total_processed // args.batch_size + 1
            ) > hparams.learning_rate_warmup_steps:
                scheduler.step(best_dev_fscore)


def run_test(args):

    test_path = args.test_ptb_path

    if args.dataset == "ctb":
        test_path = args.test_ctb_path

    print("Loading model from {}...".format(args.model_path_base))
    assert args.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

    info = torch_load(args.model_path_base)
    assert "hparams" in info["spec"], "Older savefiles not supported"
    parser = Lparser.ChartParser.from_spec(info["spec"], info["state_dict"])
    parser.eval()

    print("Loading test trees from {}...".format(test_path))
    test_treebank = trees.load_trees(test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Parsing test sentences...")
    start_time = time.time()

    punct_set = "." "``" "''" ":" ","

    parser.eval()
    test_predicted = []
    for start_index in range(0, len(test_treebank), args.eval_batch_size):
        subbatch_trees = test_treebank[start_index : start_index + args.eval_batch_size]

        subbatch_sentences = [
            [(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees
        ]

        (
            predicted,
            _,
        ) = parser.parse_batch(subbatch_sentences)
        del _
        test_predicted.extend([p.convert() for p in predicted])

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted)
    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )


def run_parse(args):

    print("Loading model from {}...".format(args.model_path_base))
    assert args.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

    info = torch_load(args.model_path_base)
    assert "hparams" in info["spec"], "Older savefiles not supported"
    parser = Lparser.ChartParser.from_spec(info["spec"], info["state_dict"])
    parser.eval()
    print("Parsing sentences...")
    with open(args.input_path) as input_file:
        sentences = input_file.readlines()
    sentences = [sentence.split() for sentence in sentences]

    # Parser does not do tagging, so use a dummy tag when parsing from raw text
    if "UNK" in parser.tag_vocab.indices:
        dummy_tag = "UNK"
    else:
        dummy_tag = parser.tag_vocab.value(0)

    start_time = time.time()

    def save_data(syntree_pred, cun):
        appent_string = "_" + str(cun) + ".txt"
        if args.output_path_synconst != "-":
            with open(args.output_path_synconst + appent_string, "w") as output_file:
                for tree in syntree_pred:
                    output_file.write("{}\n".format(tree.pred_linearize()))
            print("Output written to:", args.output_path_synconst)

    syntree_pred = []
    cun = 0
    for start_index in range(0, len(sentences), args.eval_batch_size):
        subbatch_sentences = sentences[start_index : start_index + args.eval_batch_size]

        subbatch_sentences = [
            [(dummy_tag, word) for word in sentence] for sentence in subbatch_sentences
        ]
        syntree, _ = parser.parse_batch(subbatch_sentences)
        syntree_pred.extend(syntree)
        if args.save_per_sentences <= len(syntree_pred) and args.save_per_sentences > 0:
            save_data(syntree_pred, cun)
            syntree_pred = []
            cun += 1

    if 0 < len(syntree_pred):
        save_data(syntree_pred, cun)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(args, hparams))
    hparams.populate_arguments(subparser)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--embedding-path", required=True)
    subparser.add_argument("--embedding-type", default="random")

    subparser.add_argument("--model-name", default="test")
    subparser.add_argument("--evalb-dir", default="./EVALB/")

    subparser.add_argument("--dataset", default="ptb")

    subparser.add_argument("--train-ptb-path", default="./data/ptb/02-21.10way.clean")
    subparser.add_argument("--dev-ptb-path", default="./data/ptb/22.auto.clean")

    subparser.add_argument("--train-ctb-path", default="./data/ctb/train.txt")
    subparser.add_argument("--dev-ctb-path", default="./data/ctb/dev.txt")

    subparser.add_argument("--batch-size", type=int, default=250)
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)
    subparser.add_argument("--eval-batch-size", type=int, default=30)
    subparser.add_argument("--epochs", type=int, default=150)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="./EVALB/")
    subparser.add_argument("--embedding-path", default="./data/glove/glove.gz")
    subparser.add_argument("--dataset", default="ptb")
    subparser.add_argument("--test-ptb-path", default="./data/ptb/23.auto.clean")
    subparser.add_argument("--test-ctb-path", default="./data/ctb/test.txt")
    subparser.add_argument("--eval-batch-size", type=int, default=100)

    subparser = subparsers.add_parser("parse")
    subparser.set_defaults(callback=run_parse)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--embedding-path", default="./data/ptb/glove.gz")
    subparser.add_argument("--dataset", default="ptb")
    subparser.add_argument("--save-per-sentences", type=int, default=-1)
    subparser.add_argument("--input-path", type=str, required=True)
    subparser.add_argument("--output-path-synconst", type=str, default="-")
    subparser.add_argument("--eval-batch-size", type=int, default=50)

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
