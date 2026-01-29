import os, csv, json, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from onnx2torch import convert
from captum.attr import LayerIntegratedGradients

from collections import defaultdict
import torch
import numpy as np
from heuristics.babsr import BabsrBranching
from utils import get_reduce_op


class FsbBranching(BabsrBranching):

    @torch.no_grad()
    def get_branching_decisions(
            self, domains, split_depth, branching_candidates=5,
            branching_reduceop='min', use_beta=False, prioritize_alphas='none',
            **kwargs):

        lower_bounds, upper_bounds = domains['lower_bounds'], domains['upper_bounds']
        orig_mask, lAs, cs = domains['mask'], domains['lAs'], domains['cs']
        history = domains['history']
        alphas, betas = domains['alphas'], domains['betas']
        rhs = domains['thresholds']


        # =========================
        # CONFIG
        # =========================
        ONNX_PATH = "/home/z.dehghanipour/vnncomp2024_benchmarks/benchmarks/oval21/nets/cifar_base_kw.onnx"
        IMAGE_INDEX = 0                # CIFAR-10 test image index
        N_STEPS     = 32               # IG steps (higher = smoother, slower)
        TOP_DRAW    = 40               # max nodes drawn per column in diagram (for readability)
        OUT_DIR     = "./xai_fsb_fullnodes"
        os.makedirs(OUT_DIR, exist_ok=True)

        # =========================
        # CIFAR-10 image
        # =========================


        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
        image, _ = ds[IMAGE_INDEX]
        image = image.unsqueeze(0)  # (1,3,32,32)
        classes = ds.classes

        # --- save visualisation of the lower bound ---
        plt.figure(figsize=(3.5,3.5))
        plt.imshow(denorm(image).permute(1,2,0).numpy())
        plt.axis("off")
        plt.title("Verifier Lower Bound (used as XAI input)")
        plt.savefig(os.path.join(OUT_DIR, "lower_bound_input.png"), bbox_inches="tight")
        plt.close()

        def denorm(x):
            x = x.clone().squeeze(0)
            for c in range(3):
                x[c] = x[c] * STD[c] + MEAN[c]
            return x.clamp(0, 1)

        # =========================
        # MODEL: ONNX -> Torch
        # =========================
        model = convert(ONNX_PATH)
        model.eval()

        # =========================
        # ACT SITES
        # =========================
        mods = list(model.named_modules())
        ACT_TYPES = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)
        PRE_TYPES = (nn.Conv2d, nn.Linear)

        # pair activation with nearest previous Conv/Linear and keep forward index
        pairs = []  # [{idx, act_name, act_type, pre_name, pre_type, pre_module}]
        for i, (name, m) in enumerate(mods):
            if isinstance(m, ACT_TYPES):
                pre = None
                for j in range(i-1, -1, -1):
                    pname, pm = mods[j]
                    if isinstance(pm, PRE_TYPES):
                        pre = (pname, pm)
                        break
                if pre is not None:
                    pairs.append({
                        "idx": i,
                        "act_name": name, "act_type": type(m).__name__,
                        "pre_name": pre[0], "pre_type": type(pre[1]).__name__,
                        "pre_module": pre[1],
                    })

        if not pairs:
            raise RuntimeError("No activation→(nearest Conv/Linear) pairs found. "
                            "If your model fuses activations, tell me the layer names to target.")

        # =========================
        # PREDICTION
        # =========================
        def forward(x): return model(x)
        with torch.no_grad():
            logits = forward(image)
        pred_class = int(logits.argmax())
        pred_name  = classes[pred_class] if 0 <= pred_class < len(classes) else str(pred_class)

        # Save the input image preview
        img_path = os.path.join(OUT_DIR, f"image_{IMAGE_INDEX}_pred_{pred_name}.png")
        plt.figure(figsize=(3.5,3.5))
        plt.imshow(denorm(image).permute(1,2,0).numpy())
        plt.axis("off")
        plt.title(f"Predicted: {pred_name}")
        plt.savefig(img_path, bbox_inches="tight")
        plt.close()

        # =========================
        # INTEGRATED GRADIENTS
        # =========================
        # FSB order: last→first (reverse by forward index)
        pairs.sort(key=lambda d: d["idx"], reverse=True)
        baseline = torch.zeros_like(image)

        global_rows = []
        per_act_csvs = []
        columns = []
        total_nodes = 0
        architecture_desc = []

        def safe_norm01(t):
            """Return per-element normalization to [0,1] with robust handling."""
            t = t.detach().cpu()
            tmin = float(t.min()) if t.numel() > 0 else 0.0
            tmax = float(t.max()) if t.numel() > 0 else 0.0
            den  = (tmax - tmin)
            if not np.isfinite(den) or abs(den) < 1e-12:
                return torch.zeros_like(t)
            return (t - tmin) / den

        for fsb_col, pair in enumerate(pairs):
            pre_name, pre_module = pair["pre_name"], pair["pre_module"]
            pre_type, act_name, act_type = pair["pre_type"], pair["act_name"], pair["act_type"]

            lig = LayerIntegratedGradients(forward, pre_module)
            attr = lig.attribute(inputs=image, baselines=baseline, target=pred_class, n_steps=N_STEPS)

            if pre_type == "Linear":
                A = attr.squeeze(0)         # (F,)
                imp = A.abs().reshape(-1)   # (F,)
                node_count = int(imp.numel())
                architecture_desc.append(f"Linear({pre_module.in_features}, {pre_module.out_features})")
                # flat -> index only
                indices = [(int(i), None, None) for i in range(node_count)]
                node_kind = "linear"
            else:
                A = attr.squeeze(0)         # (C,H,W)
                C, H, W = A.shape
                imp = A.abs().reshape(-1)   # (C*H*W,)
                node_count = int(imp.numel())
                ksize = pre_module.kernel_size
                ksize = ksize[0] if isinstance(ksize, tuple) else int(ksize)
                architecture_desc.append(f"Conv({pre_module.in_channels}, {pre_module.out_channels}, {ksize})")
                # map flat -> (c,h,w)
                indices = [(c, h, w) for c in range(C) for h in range(H) for w in range(W)]
                node_kind = "conv"

            total_nodes += node_count

            imp_cpu  = imp.detach().cpu()
            imp_norm = safe_norm01(imp_cpu)  # [0,1]

            # sort by raw score
            sorted_idx = torch.argsort(imp_cpu, descending=True)
            vals_sorted = imp_cpu[sorted_idx]
            kplot = min(TOP_DRAW, node_count)

            top_idx  = sorted_idx[:kplot]
            top_vals = vals_sorted[:kplot]
            top_norm = imp_norm[top_idx]

            rem_idx = sorted_idx[kplot:]
            if rem_idx.numel() > 0:
                others_norm  = float(imp_norm[rem_idx].mean())
                others_count = int(rem_idx.numel())
            else:
                others_norm  = None
                others_count = 0

            # ---------- CSV writing ----------
            csv_path = os.path.join(OUT_DIR, f"FSB_{fsb_col:02d}__pre_{pre_name.replace('.','_')}__act_{act_name.replace('.','_')}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                if pre_type == "Linear":
                    writer.writerow(["rank","pre_layer","pre_type","act_layer","act_type",
                                    "neuron_index","score_raw","score_norm_0_1"])
                    for r, flat_i in enumerate(sorted_idx.tolist()):
                        writer.writerow([r, pre_name, pre_type, act_name, act_type,
                                        flat_i, float(imp_cpu[flat_i]), float(imp_norm[flat_i])])
                        global_rows.append({
                            "fsb_col": fsb_col, "rank_local": r, "pre_layer": pre_name, "pre_type": pre_type,
                            "act_layer": act_name, "act_type": act_type,
                            "node_kind": "linear", "index": flat_i,
                            "index_c": "", "index_h": "", "index_w": "",
                            "score_raw": float(imp_cpu[flat_i]),
                            "score_norm_0_1": float(imp_norm[flat_i]),
                            "node_count_in_layer": node_count,
                        })
                else:
                    writer.writerow(["rank","pre_layer","pre_type","act_layer","act_type",
                                    "c","h","w","flat_index","score_raw","score_norm_0_1"])
                    for r, flat_i in enumerate(sorted_idx.tolist()):
                        c, h, w = indices[flat_i]
                        writer.writerow([r, pre_name, pre_type, act_name, act_type,
                                        c, h, w, flat_i, float(imp_cpu[flat_i]), float(imp_norm[flat_i])])
                        global_rows.append({
                            "fsb_col": fsb_col, "rank_local": r, "pre_layer": pre_name, "pre_type": pre_type,
                            "act_layer": act_name, "act_type": act_type,
                            "node_kind": "conv", "index": flat_i,
                            "index_c": c, "index_h": h, "index_w": w,
                            "score_raw": float(imp_cpu[flat_i]),
                            "score_norm_0_1": float(imp_norm[flat_i]),
                            "node_count_in_layer": node_count,
                        })
            per_act_csvs.append(csv_path)





        # [MAX]
        # print("+ Unstable Neurons (mask):  ", sum([torch.sum(m).item() for m in orig_mask.values()]))
        
        # print("+ Unstable Neurons (bounds):", sum([(torch.sign(l) != torch.sign(u)).sum().item() 
        #                                            for l, u in zip(lower_bounds.values(), upper_bounds.values())]))
        # print("+", history)

        batch = len(next(iter(orig_mask.values())))
        # Mask is 1 for unstable neurons. Otherwise it's 0.
        mask = orig_mask
        reduce_op = get_reduce_op(branching_reduceop)
        # In case number of unstable neurons less than topk
        topk = min(branching_candidates,
                   int(sum([item.sum() for item in mask.values()]).item()))
        number_bounds = 1 if cs is None else cs.shape[1]
        score, intercept_tb = self.babsr_score(
            lower_bounds, upper_bounds, lAs, mask, reduce_op,
            number_bounds, prioritize_alphas)

        final_decision = [[] for _ in range(batch)]
        decision_tmp = {}
        tmp_ret = {}
        score_from_layer_idx = 1 if len(score) > 1 else 0
        skip_layers = list(range(score_from_layer_idx))

        # real batch = batch * 2, since we have two kinds of scores
        lbs = {k: torch.concat([v, v]) for k, v in lower_bounds.items()}
        ubs = {k: torch.concat([v, v]) for k, v in upper_bounds.items()}
        # per neuron alpha.
        sps = defaultdict(dict)
        for k, vv in alphas.items():
            sps[k] = {}
            for kk, v in vv.items():
                sps[k][kk] = torch.cat([v, v], dim=2)
        if use_beta:
            bs = [torch.cat([i, i]) for i in betas]
            history += history
        rhs = torch.cat([rhs, rhs])
        if cs is not None:
            cs = torch.cat([cs, cs])
        set_alpha = True  # We only set the alpha once.

        for i in range(score_from_layer_idx, len(score)):
            if ((score[i].max(1).values <= 1e-4).all()
                    and (intercept_tb[i].min(1).values >= -1e-4).all()):
                print(f'{i}th layer has no valid scores')
                skip_layers.append(i)
                continue
            score_idx = torch.topk(score[i], topk)
            score_idx_indices = score_idx.indices.cpu()
            itb_idx = torch.topk(intercept_tb[i], topk, largest=False)
            itb_idx_indices = itb_idx.indices.cpu()
            k_ret = torch.empty(size=(topk, batch * 2), device=score[i].device)
            k_decision = []
            for k in range(topk):
                decision_index = score_idx_indices[:, k]
                # add decision_index with layer's idx
                decision_max_ = [[i, j.item()] for j in decision_index]
                decision_index = itb_idx_indices[:, k]
                decision_min_ = [[i, j.item()] for j in decision_index]
                k_decision.append(decision_max_ + decision_min_)
                # only save the best lower bounds of the two splits
                args_update_bounds = {
                    'lower_bounds': lbs, 'upper_bounds': ubs,
                    'alphas': sps if set_alpha else {}, 'cs': cs
                }
                split = {'decision': k_decision[-1]}
                self.net.build_history_and_set_bounds(args_update_bounds, split)
                if use_beta:
                    args_update_bounds.update({'betas': bs, 'history': history})
                    k_ret_lbs = self.net.update_bounds(
                        args_update_bounds,
                        fix_interm_bounds=True, shortcut=True, beta_bias=False)
                else:
                    k_ret_lbs = self.net.update_bounds(
                        args_update_bounds, beta=False,
                        fix_interm_bounds=True, shortcut=True, beta_bias=False)
                # consider the max improvement among multi bounds in one C matrix
                k_ret_lbs = (k_ret_lbs - torch.cat([rhs, rhs])).max(-1).values
                # No need to set alpha next time; we do not optimize the alphas.
                set_alpha = False
                # build mask indicates invalid scores (stable neurons), batch wise, 1: invalid
                mask_score = (score_idx.values[:, k] <= 1e-4).float()
                mask_itb = (itb_idx.values[:, k] >= -1e-4).float()
                # make the invalid lower bounds worse than normal lower bounds by minus 999999
                # we only consider the best lower bound across two splits by using min(0)
                k_ret[k] = reduce_op((
                    k_ret_lbs.view(-1) - torch.cat(
                        [mask_score, mask_itb]).repeat(2) * 999999
                    ).reshape(2, -1),
                    dim=0).values
            split_depth = min(split_depth, k_ret.shape[0])
            i_idx = k_ret.topk(split_depth, dim=0)  # compare across topK
            tmp_ret[i] = i_idx.values  # [split_depth, batch*2]
            tmp_indice = i_idx.indices
            decision_tmp[i] = [
                k_decision[tmp_indice[ii // (2 * batch)][
                    ii % (2 * batch)]][ii % (2 * batch)]
                for ii in range(split_depth * (batch * 2))
            ]

        # shape of tmp_ret: [layer, num_split, batch*2]
        if len(tmp_ret):
            stacked_layers = torch.stack([i for i in tmp_ret.values()])  # [layer, split_depth, batch*2]
            max_ret = torch.topk(stacked_layers.view(-1, batch * 2), split_depth,
                                 dim=0)  # compare across layers [split_depth, batch*2]
            # shape: [num_split*batch*2]
            rets, decision_layers = max_ret.values.view(-1).cpu().numpy(), max_ret.indices.view(
                -1).cpu().numpy()  # first batch: score; second batch: intercept_tb.
            decision_layers = decision_layers // split_depth

            # add index number for the skipped layers
            # for _, g in groupby(enumerate(skip_layers), lambda ix: ix[0] - ix[1]):
            #     decision_layers[decision_layers >= list(g)[-1][-1]] += 1
            for s in skip_layers:
                decision_layers[decision_layers >= s] += 1

            for l in range(split_depth):
                for b in range(batch):
                    decision_layer_1, decision_index_1 = decision_tmp[
                        decision_layers[2 * l * batch + b].item()][
                        l * 2 * batch + b]
                    decision_layer_2, decision_index_2 = decision_tmp[
                        decision_layers[2 * l * batch + b + batch].item()
                    ][l * 2 * batch + b + batch]
                    decision_layer_1 = self.net.split_nodes[decision_layer_1].name
                    decision_layer_2 = self.net.split_nodes[decision_layer_2].name
                    len_final_decision = len(final_decision[b])
                    if (max([s[b].max() for s in score]) > 1e-4
                            and min([s[b].min() for s in intercept_tb]) < -1e-4
                            and max(rets[2 * l * batch + b], rets[2 * l * batch + b + batch]) > -10000
                            and (mask[decision_layer_1][b][decision_index_1] != 0
                                 or mask[decision_layer_2][b][decision_index_2] != 0)
                    ):  # make sure this potential split is valid
                        if (rets[2 * l * batch + b] > rets[2 * l * batch + b + batch]
                                and mask[decision_layer_1][b][decision_index_1] != 0):  # score > intercept_tb
                            final_decision[b].append(
                                decision_tmp[decision_layers[2 * l * batch + b].item()][l * 2 * batch + b])
                        elif mask[decision_layer_2][b][decision_index_2] != 0:
                            final_decision[b].append(decision_tmp[decision_layers[2 * l * batch + b + batch].item()][
                                                         l * 2 * batch + b + batch])
                        else:
                            mask_item = {k: m[b] for k, m in mask.items()}
                            for preferred_layer in np.random.choice(len(self.net.split_nodes), len(self.net.split_nodes), replace=False):
                                preferred_layer_ = self.net.split_nodes[preferred_layer].name
                                if len(mask_item[preferred_layer_].nonzero(as_tuple=False)) != 0:
                                    final_decision[b].append(
                                        [preferred_layer, mask_item[preferred_layer_].nonzero(as_tuple=False)[0].item()])
                                    break
                    else:
                        # using a random choice
                        mask_item = {k: m[b] for k, m in mask.items()}
                        for preferred_layer in np.random.choice(len(self.net.split_nodes), len(self.net.split_nodes), replace=False):
                            preferred_layer_ = self.net.split_nodes[preferred_layer].name
                            if len(mask_item[preferred_layer_].nonzero(as_tuple=False)) != 0:
                                final_decision[b].append(
                                    [preferred_layer, mask_item[preferred_layer_].nonzero(as_tuple=False)[0].item()])
                                break
                    if len(final_decision[b]) > len_final_decision:
                        final_decision_ = self.net.split_nodes[final_decision[b][-1][0]].name
                        mask[final_decision_][b][final_decision[b][-1][1]] = 0
        else:
            # all layers are split or has no improvement
            for b in range(split_depth * batch):
                # using a random choice
                mask_item = {k: m[b] for k, m in mask.items()}

                for preferred_layer in reversed(self.net.split_nodes):
                    layer_name = preferred_layer.name
                    if len(mask_item[layer_name].nonzero(as_tuple=False)) != 0:
                        final_decision[b].append(
                            [self.net.split_nodes.index(preferred_layer), mask_item[layer_name].nonzero(as_tuple=False)[0].item()])

        split_depth = min([len(d) for d in final_decision])
        final_decision = [[batch[i] for batch in final_decision] for i in
                          range(split_depth)]  # change the order of final decision to split_depth * batch
        final_decision = sum(final_decision, [])


        return final_decision, None, split_depth # None for points




