import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import Dataset_Codec_Antispoofing, genSpoof_list
from evaluation import calculate_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    config["database_path"] = args.database_path

    if not args.evalcodecname:
        # NOTE: for training
        codecname = args.traincodecname
        print(f"Training on Codec {codecname}")
    else:
        # NOTE: for evaluation
        codecname = args.evalcodecname
        print(f"Eval on Codec {codecname}")

    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    filelist_path = Path(config["database_path"])

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        codecname,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"],
        config["batch_size"],
    )
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    if args.eval:
        model_tag = model_tag / "eval"
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    # if device == "cpu":
    #     raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        filelist_path, args.seed, config, codec_name=codecname, eval=args.eval
    )

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Evaluating ckpt at: {}".format(args.model_path))
        eval_score_path = "{}/eval_results/{}/{}".format(
            os.path.dirname(args.model_path), codecname, config["eval_output"]
        )
        produce_evaluation_file(eval_loader, model, device, eval_score_path)
        calculate_EER(cm_scores_file=eval_score_path, output_file=model_tag / "EER.txt")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 100.0
    best_eval_eer = 100.0
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(
            trn_loader, model, optimizer, device, scheduler, config
        )
        produce_evaluation_file(
            dev_loader, model, device, metric_path / "dev_score.txt"
        )
        dev_eer = calculate_EER(
            cm_scores_file=metric_path / "dev_score.txt",
            output_file=metric_path / "dev_EER_{}epo.txt".format(epoch),
            printout=False,
        )
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}".format(running_loss, dev_eer))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)

        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(
                model.state_dict(),
                model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer),
            )

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(eval_loader, model, device, eval_score_path)
                eval_eer = calculate_EER(
                    cm_scores_file=eval_score_path,
                    output_file=metric_path / "EER_{:03d}epo.txt".format(epoch),
                )

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    produce_evaluation_file(eval_loader, model, device, eval_score_path)
    eval_eer = calculate_EER(
        cm_scores_file=eval_score_path, output_file=model_tag / "EER.txt"
    )
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}".format(eval_eer))
    f_log.close()

    torch.save(model.state_dict(), model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
        torch.save(model.state_dict(), model_save_path / "best.pth")
    print("Exp FIN. EER: {:.3f}".format(best_eval_eer))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
    filelist_path: str,
    seed: int,
    config: dict,
    codec_name: str = None,
    eval: bool = False,
) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""

    if eval:
        trn_loader = None
        dev_loader = None
    else:
        d_label_trn, file_train, base_dir, others = genSpoof_list(
            dir_meta=filelist_path, codec_name=codec_name, is_train=True, is_eval=False
        )
        print("no. training files:", len(file_train))

        train_set = Dataset_Codec_Antispoofing(
            list_IDs=file_train, labels=d_label_trn, base_dir=base_dir, others=others
        )
        gen = torch.Generator()
        gen.manual_seed(seed)
        trn_loader = DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=gen,
        )

        d_label_dev, file_dev, base_dir, others = genSpoof_list(
            dir_meta=filelist_path, codec_name=codec_name, is_train=False, is_eval=False
        )
        print("no. validation files:", len(file_dev))

        dev_set = Dataset_Codec_Antispoofing(
            list_IDs=file_dev, labels=d_label_dev, base_dir=base_dir, others=others
        )
        dev_loader = DataLoader(
            dev_set,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    d_label_eval, file_eval, base_dir, others = genSpoof_list(
        dir_meta=filelist_path, codec_name=codec_name, is_train=False, is_eval=True
    )
    print("no. evaluation files:", len(file_eval))
    eval_set = Dataset_Codec_Antispoofing(
        list_IDs=file_eval, labels=d_label_eval, base_dir=base_dir, others=others
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader, model, device: torch.device, save_path: str
) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    fname_list = []
    score_list = []
    label_list = []
    others_list = []
    for batch_x, label, utt_id, others in tqdm(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        label_list.extend(label.tolist())
        score_list.extend(batch_score.tolist())

    assert len(fname_list) == len(score_list)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w") as fh:
        for fname, score, label in zip(fname_list, score_list, label_list):
            fh.write("{} {} {}\n".format(fname, score, label))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace,
):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.5, 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y, _, _ in tqdm(trn_loader, desc="train"):
        ## Note: batch_x [torch.Tensor(float)]: (batch, 64600); batch_y [torch.Tensor(int)]:(batch, )
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument(
        "--database_path",
        dest="database_path",
        type=str,
        help="list of audio files",
        default="datalist.csv",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        help="configuration file",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed (default: 1234)"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit",
    )
    parser.add_argument(
        "--comment", type=str, default=None, help="comment to describe the saved model"
    )
    parser.add_argument(
        "--eval_model_weights",
        type=str,
        default=None,
        help="directory to the model weight file (can be also given in the config file)",
    )
    parser.add_argument(
        "--traincodecname",
        type=str,
        default="SpeechTokenizer",
        help="the training codec",
    )
    parser.add_argument(
        "--evalcodecname",
        type=str,
        default=None,
        help="the evaluation codec",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="the evaluation codec",
    )
    main(parser.parse_args())
