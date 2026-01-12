import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

# sys.path.insert(0, 'utils/')
# sys.path.insert(0, 'model/')
from model.disco import DISCO
from losses.disco_loss import DiscoLoss
from utils import DiscoDataset, save_object, calc_mode, D_KL, get_params, read_data
from sklearn.metrics import classification_report, f1_score
import argparse
import wandb
from wandb_creds import wandb_creds 
os.environ["WANDB_API_KEY"] = wandb_creds()
"""
    Trains a prototype label distributional learning neural model (LDL-NM) which
    is an artificial neural network that jointly learns to model label distributions
    for ground truth labels, items, and annotators. The resultant model can be
    used to iteratively infer embeddings for annotators (or be used to
    conduct majority/modal voting across its memory of known annotators).

    Here is an example run `python3 train_disco_sweep.py --config ./config_files/disco_config.cfg --sweep_id rit_pl/jobq2_sweep/smd8rl2y --gpu_id 8`

    In order to run this, you initially need to create a Sweep on Weights and Biases. 
"""

def wandb_logging_dev(disco_model_params,epoch,agg_acc, KLi, dev_agg_acc, dev_KLi, f1_macro, dev_f1_macro, precision_macro, dev_precision_macro, recall_macro, dev_recall_macro):
    wandb_log = {
        "train KL": KLi,
        "train F1": f1_macro,
        "train Accuracy": agg_acc,
        "train precision": precision_macro,
        "train recall": recall_macro,
        "dev KL": dev_KLi,
        "dev F1": dev_f1_macro,
        "dev Accuracy": dev_agg_acc,
        "dev precision": dev_precision_macro,
        "dev recall": dev_recall_macro,
        "epoch": epoch,
        "dataset": disco_model_params['dataset']
        }

    # logging accuracy
    wandb.log(wandb_log)   


def wandb_logging_train(disco_model_params,epoch,agg_acc, KLi, f1_macro, precision_macro, recall_macro):
    wandb_log = {
        "train KL": KLi,
        "train F1": f1_macro,
        "train Accuracy": agg_acc,
        "train precision": precision_macro,
        "train recall": recall_macro,
        "epoch": epoch,
        "dataset": disco_model_params['dataset']
        }

    # logging accuracy
    wandb.log(wandb_log)   

def split(design_mat, n_valid=10):  # a simple design matrix splitting function if needed
    valid_mat = design_mat[0:n_valid, :]
    train_mat = design_mat[n_valid:design_mat.shape[0], :]
    return valid_mat, train_mat


def calc_stats(model, loss_fn, Xi_, Yi_, Ya_, Y_, A_, I_, batch_size, agg_type="mode",
               n_subset=1000, eval_aggreg=True):
    """
        Calculates fixed-point statistics, i.e., accuracy and cost
    """
    dataset = DiscoDataset(Xi_, Yi_, Ya_, Y_, A_)
    if n_subset > 0:
        ptrs = np.random.permutation(len(dataset))[0:n_subset]
        dataset = Subset(dataset, ptrs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    L = 0.0
    KLi = 0.0
    KLa = 0.0
    agg_KL = 0.0
    acc = 0.0
    agg_acc = 0.0  # aggregated accuracy
    device = next(model.parameters()).device
    
    # Collect all predictions and ground truth labels
    all_y_pred = []
    all_y_test = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            a_s = batch["a"].to(device)
            xi_s = batch["xi"].to(device)
            y_s = batch["y"].to(device)
            y_ind = torch.argmax(y_s, dim=1).to(torch.int64)
            yi_s = batch["yi"].to(device)
            ya_s = batch["ya"].to(device)

            y_logits, yi_logits, ya_logits = model(xi_s, a_s)
            pY = torch.softmax(y_logits, dim=-1)
            Ns = y_s.shape[0]
            L_t, metrics = loss_fn(y_logits, yi_logits, ya_logits, y_s, yi_s, ya_s, model=model)
            L += L_t.item() * Ns
            KLi += metrics["Lyi"].item() * Ns
            KLa += metrics["Lya"].item() * Ns

            # compute accuracy of predictions
            y_pred = torch.argmax(pY, dim=1).to(torch.int64)
            comp = (y_pred == y_ind).to(torch.float32)
            acc += torch.sum(comp).item()

            # Collect predictions and ground truth for metrics calculation
            all_y_pred.append(y_pred)
            all_y_test.append(y_ind)

            # compute aggregated accuracy across internally known annotators
            sub_acc = 0.0

            if eval_aggreg is True:
                for s in range(xi_s.shape[0]):
                    xs = xi_s[s, :].unsqueeze(0)
                    ys = y_s[s, :].unsqueeze(0)
                    ys_ind = torch.argmax(ys, dim=1).to(torch.int64)
                    py, _ = model.decode_y_ensemble(xs)
                    y_label_preds = torch.mean(py, dim=0, keepdim=True)
                    agg_KL += D_KL(ys, y_label_preds).item()
                    if agg_type == "mode":
                        yhat_set = torch.argmax(py, dim=1).cpu().tolist()
                        y_mode, y_freq = calc_mode(yhat_set)  # compute mode of predictions
                        comp = 1.0 if y_mode == ys_ind.item() else 0.0
                        sub_acc += comp
                    else:  # == "expectation"
                        y_mean = torch.mean(py, dim=0, keepdim=True)
                        y_pred_agg = torch.argmax(y_mean, dim=1).to(torch.int64)
                        comp = 1.0 if y_pred_agg.item() == ys_ind.item() else 0.0
                        sub_acc += comp

            agg_acc += sub_acc

    total_samples = len(dataset)
    acc = acc / (total_samples * 1.0)
    L = L / (total_samples * 1.0)
    KLi = KLi / (total_samples * 1.0)
    KLa = KLa / (total_samples * 1.0)
    agg_KL = agg_KL / (total_samples * 1.0)
    agg_acc = agg_acc / (total_samples * 1.0)
    # classification report using collected predictions and ground truth
    y_test = torch.cat(all_y_test, dim=0)
    y_pred = torch.cat(all_y_pred, dim=0)
    f1_macro = f1_score(y_test.cpu().numpy(), y_pred.cpu().numpy(), average='macro')

    f1_micro = f1_score(y_test.cpu().numpy(), y_pred.cpu().numpy(), average='micro')

    f1_weighted = f1_score(y_test.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

    results = classification_report(y_test.cpu().numpy(), y_pred.cpu().numpy(), digits=3, output_dict=True)
    precision_macro = results['macro avg']['precision']
    precision_weighted = results['weighted avg']['precision']

    recall_macro = results['macro avg']['recall']
    recall_weighted = results['weighted avg']['recall']

    return acc, L, KLi, KLa, agg_acc, f1_macro, f1_micro, f1_weighted, precision_macro, precision_weighted, recall_macro, recall_weighted, agg_KL


def _build_config(data, disco_model_params):
    return {
        "xi_dim": data["n_xi"],
        "yi_dim": data["yi_dim"],
        "ya_dim": data["ya_dim"],
        "y_dim": data["y_dim"],
        "a_dim": data["n_a"],
        "lat_dim": disco_model_params["lat_dim"],
        "act_fx": disco_model_params["act_fx"],
        "init_type": disco_model_params["weight_init_scheme"],
        "lat_i_dim": disco_model_params["lat_i_dim"],
        "lat_a_dim": disco_model_params["lat_a_dim"],
        "lat_fusion_type": disco_model_params["lat_fusion_type"],
        "drop_p": disco_model_params["drop_p"],
        "gamma_i": disco_model_params["gamma_i"],
        "gamma_a": disco_model_params["gamma_a"],
        "l1_norm": disco_model_params.get("l1_norm", 0.0),
        "l2_norm": disco_model_params.get("l2_norm", 0.0),
        "name": disco_model_params.get("name"),
        "i_dim": disco_model_params.get("i_dim"),
    }



def train_disco(data, simulation_params, disco_model_params, params):
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    model = DISCO(xi_dim=data["n_xi"], yi_dim=data["yi_dim"], ya_dim=data["ya_dim"], y_dim=data["y_dim"],
                  a_dim=data["n_a"],
                  lat_dim=disco_model_params["lat_dim"], act_fx=disco_model_params["act_fx"],
                  init_type=disco_model_params["weight_init_scheme"],
                  lat_i_dim=disco_model_params["lat_i_dim"], lat_a_dim=disco_model_params["lat_a_dim"],
                  lat_fusion_type=disco_model_params["lat_fusion_type"],
                  drop_p=disco_model_params["drop_p"], gamma_i=disco_model_params["gamma_i"],
                  gamma_a=disco_model_params["gamma_a"])
    model.to(device)
    loss_fn = DiscoLoss(gamma_i=disco_model_params["gamma_i"], gamma_a=disco_model_params["gamma_a"])
    optimizer = torch.optim.Adam(model.parameters(), lr=disco_model_params["learning_rate"])

    # Z = model.encode(Xi, A)
    # gen_data_plot(Z, Y, use_tsne=False, fname="latents")

    ################################################################################
    # fit model to the design matrices
    ################################################################################

    # wandb_initialize(disco_model_params,simulation_params["n_epoch"])
    acc, L, KLi, KLa, agg_acc, f1_macro, f1_micro, f1_weighted, precision_macro, precision_weighted, recall_macro, recall_weighted, train_agg_KL = calc_stats(model, loss_fn, data["Xi"], data["Yi"], data["Ya"], data["Y"], data["A"], data["I"],
                                           simulation_params["batch_size"])
    if data["dev_Y"] is not None:

        dev_acc, dev_L, dev_KLi, dev_KLa, dev_agg_acc, dev_f1_macro, dev_f1_micro, dev_f1_weighted, dev_precision_macro, dev_precision_weighted, dev_recall_macro, dev_recall_weighted,dev_agg_KL = calc_stats(model, loss_fn, data["dev_Xi"], data["dev_Yi"],
                                                                   data["dev_Ya"], data["dev_Y"], data["dev_A"],
                                                                   data["dev_I"], simulation_params["batch_size"])

        print(
            " {0}: Fit.Acc = {1} E.Acc = {2} L = {3} | Dev.Acc = {4} E.Acc = {5}  KL = {6} ".format(-1, acc, agg_acc,
                                                                                                    dev_acc, dev_L,
                                                                                                    dev_agg_acc,
                                                                                                    dev_agg_KL))
    else:
        print(
            " {0}: Fit.Acc = {1}  E.Acc = {2} L = {3}  KLi = {4}  KLa = {5}".format(-1, acc, agg_acc, L, KLi, KLa))
    simulation_params["n_epoch"] = 5
    train_dataset = data.get("train_dataset") or DiscoDataset(
        data["Xi"],
        data["Yi"],
        data["Ya"],
        data["Y"],
        data["A"],
    )
    train_loader = DataLoader(train_dataset, batch_size=simulation_params["batch_size"], shuffle=True)
    for e in range(simulation_params["n_epoch"]):
        L = 0.0  # epoch loss
        Ns = 0.0
        model.train()
        for batch in train_loader:
            a_s = batch["a"].to(device)
            y_s = batch["y"].to(device)
            xi_s = batch["xi"].to(device)
            yi_s = batch["yi"].to(device)
            ya_s = batch["ya"].to(device)

            # update model parameters and track approximate training loss
            optimizer.zero_grad(set_to_none=True)
            y_logits, yi_logits, ya_logits = model(xi_s, a_s)
            L_t, _ = loss_fn(y_logits, yi_logits, ya_logits, y_s, yi_s, ya_s, model=model)
            L_t.backward()
            if disco_model_params["update_radius"] > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=disco_model_params["update_radius"])
            optimizer.step()
            L = (L_t.item() * y_s.shape[0]) + L
            Ns += y_s.shape[0]
            print("\r{0}: L = {1}  ({2} samples seen)".format(e, (L / Ns), Ns), end="")
        print()
        if e % simulation_params["eval_every"] == 0:
            acc, L, KLi, KLa, agg_acc, f1_macro, f1_micro, f1_weighted, precision_macro, precision_weighted, recall_macro, recall_weighted, train_agg_KL = calc_stats(model, loss_fn, data["Xi"], data["Yi"], data["Ya"], data["Y"], data["A"],
                                                   data["I"], simulation_params["batch_size"])
            if data["dev_Y"] is not None:
                dev_acc, dev_L, dev_KLi, dev_KLa, dev_agg_acc, dev_f1_macro, dev_f1_micro, dev_f1_weighted, dev_precision_macro, dev_precision_weighted, dev_recall_macro, dev_recall_weighted, dev_agg_KL = calc_stats(model, loss_fn, data["dev_Xi"], data["dev_Yi"],
                                                                           data["dev_Ya"], data["dev_Y"], data["dev_A"],
                                                                           data["dev_I"],
                                                                           simulation_params["batch_size"])
                wandb_logging_dev(params,e,agg_acc, train_agg_KL, dev_agg_acc, dev_agg_KL, f1_macro, dev_f1_macro, precision_macro, dev_precision_macro, recall_macro, dev_recall_macro)
                print(" {0}: Fit.Acc = {1} E.Acc = {2} L = {3} | Dev.Acc = {4} E.Acc = {5}  KL = {6} ".format(e, acc,
                                                                                                              agg_acc,
                                                                                                              dev_acc,
                                                                                                              dev_L,
                                                                                                              dev_agg_acc,
                                                                                                              dev_agg_KL))
            else:
                wandb_logging_train(params,e,agg_acc, train_agg_KL, f1_macro, precision_macro, recall_macro)
                print(
                    " {0}: Fit.Acc = {1}  E.Acc = {2} L = {3}  KLi = {4}  KLa = {5}".format(e, acc, agg_acc, L, KLi,
                                                                                            KLa))
        if e % simulation_params["save_every"] == 0:  # save a checkpoint model
            save_object(model, "{0}/trained_model.disco".format(params["out_dir"]), config=_build_config(data, disco_model_params))

    ################################################################################
    # save final model to disk
    ################################################################################
    save_object(model, "{0}/trained_model.disco".format(params["out_dir"]), config=_build_config(data, disco_model_params))
    if data["dev_Y"] is not None:
        wandb_logging_dev(params,e,agg_acc, train_agg_KL, dev_agg_acc, dev_agg_KL, f1_macro, dev_f1_macro, precision_macro, dev_precision_macro, recall_macro, dev_recall_macro)
    else:
        wandb_logging_train(params,e,agg_acc, train_agg_KL, f1_macro, precision_macro, recall_macro)

def read_wandb_sweep_id(sweep_id, params, simulation_params, run_count):
    data = read_data(params)
    def train(config=None):
        with wandb.init(config=config):
            disco_model_params = wandb.config
            train_disco(data, simulation_params, disco_model_params, params)

    wandb.agent(sweep_id, train, count=run_count)


def main():
    ################################################################################
    # read in configuration file and extract necessary variables/constants
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file")
    parser.add_argument("--sweep_id", help="Sweep ID from WANDB")
    parser.add_argument("--gpu_id", help="GPU id",default=-1)
    parser.add_argument("--run_count", help="Run Count",default=10)

    args = parser.parse_args()
    cfg_fname = args.config
    sweep_id = args.sweep_id

    gpu_id = int(args.gpu_id)
    run_count = int(args.run_count)
    if gpu_id>-1:
        print(" > Using GPU ID {0}".format(gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params, simulation_params, disco_model_params = get_params(cfg_fname)
    if params.get("seed") is not None:
        torch.manual_seed(params["seed"])
        np.random.seed(params["seed"])
    read_wandb_sweep_id(sweep_id, params, simulation_params, run_count)



if __name__ == '__main__':
    main()
