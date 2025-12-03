import time
from collections import namedtuple
from typing import Literal

import gurobipy as grb
import torch
import torch.nn as nn
from lp_mip_solver import copy_model, update_model_bounds

LPVars = namedtuple("LPVars", ["primal_input", "primals", "duals"])


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class View(nn.Module):
    """
    This is necessary in order to reshape "flat activations" such as used by
    nn.Linear with those that comes from MaxPooling
    """

    def __init__(self, out_shape):
        super(View, self).__init__()
        self.out_shape = out_shape

    def forward(self, inp):
        # We make the assumption that all the elements in the tuple have
        # the same batchsize and need to be brought to the same size

        # We assume that the first dimension is the batch size
        batch_size = inp.size(0)
        out_size = (batch_size,) + self.out_shape
        out = inp.view(out_size)
        return out


def simplify_network(all_layers):
    """
    Given a sequence of Pytorch nn.Module `all_layers`,
    representing a feed-forward neural network,
    merge the layers when two sucessive modules are nn.Linear
    and can therefore be equivalenty computed as a single nn.Linear
    """
    new_all_layers = [all_layers[0]]
    for layer in all_layers[1:]:
        if (type(layer) is nn.Linear) and (type(new_all_layers[-1]) is nn.Linear):
            # We can fold together those two layers
            prev_layer = new_all_layers.pop()

            joint_weight = torch.mm(layer.weight.data, prev_layer.weight.data)
            if prev_layer.bias is not None:
                joint_bias = layer.bias.data + torch.mv(
                    layer.weight.data, prev_layer.bias.data
                )
            else:
                joint_bias = layer.bias.data

            joint_out_features = layer.out_features
            joint_in_features = prev_layer.in_features

            joint_layer = nn.Linear(joint_in_features, joint_out_features)
            joint_layer.bias.data.copy_(joint_bias)
            joint_layer.weight.data.copy_(joint_weight)
            new_all_layers.append(joint_layer)
        elif (
            (type(layer) is nn.MaxPool1d)
            and (layer.kernel_size == 1)
            and (layer.stride == 1)
        ):
            # This is just a spurious Maxpooling because the kernel_size is 1
            # We will do nothing
            pass
        elif (type(layer) is View) and (type(new_all_layers[-1]) is View):
            # No point in viewing twice in a row
            del new_all_layers[-1]

            # Figure out what was the last thing that imposed a shape
            # and if this shape was the proper one.
            prev_layer_idx = -1
            lay_nb_dim_inp = 0
            while True:
                parent_lay = new_all_layers[prev_layer_idx]
                prev_layer_idx -= 1
                if type(parent_lay) is nn.ReLU:
                    # Can't say anything, ReLU is flexible in dimension
                    continue
                elif type(parent_lay) is nn.Linear:
                    lay_nb_dim_inp = 1
                    break
                elif type(parent_lay) is nn.MaxPool1d:
                    lay_nb_dim_inp = 2
                    break
                else:
                    raise NotImplementedError
            if len(layer.out_shape) != lay_nb_dim_inp:
                # If the View is actually necessary, add the change
                new_all_layers.append(layer)
                # Otherwise do nothing
        else:
            new_all_layers.append(layer)
    return new_all_layers


def add_single_prop(layers, ground_truth, cls):
    """
    ground_truth: ground truth lable
    class_: class we want to verify against
    """
    additional_lin_layer = torch.nn.Linear(10, 1, bias=True)
    lin_weights = additional_lin_layer.weight.data
    lin_weights.fill_(0)
    lin_bias = additional_lin_layer.bias.data
    lin_bias.fill_(0)
    lin_weights[0, cls] = -1
    lin_weights[0, ground_truth] = 1

    if torch.cuda.is_available():
        additional_lin_layer.cuda()

    final_layers = [layers[-1], additional_lin_layer]
    final_layer = simplify_network(final_layers)
    verif_layers = layers[:-1] + final_layer
    for layer in verif_layers:
        for p in layer.parameters():
            p.requires_grad = False

    return verif_layers


def update_and_solve(
    m,
    mask: list[torch.Tensor],
    lower_bounds: list[torch.Tensor],
    upper_bounds: list[torch.Tensor],
    rhs,
) -> tuple[Literal["safe", "unsafe"], torch.Tensor, LPVars]:
    """Based on `all_node_split_LP` function in `complete_verifier/lp_mip_solver.py`"""

    m.new_model: grb.Model = copy_model(m.net.solver_model, basis=False)

    pre_relu_layer_names = [relu_layer.inputs[0].name for relu_layer in m.net.relus]
    relu_layer_names = [relu_layer.name for relu_layer in m.net.relus]
    m.new_model = update_model_bounds(
        m.new_model,
        lower_bounds,
        upper_bounds,
        pre_relu_layer_names,
        relu_layer_names,
        m.net.solver_model_type,
    )
    print("Finished updating the Gurobi model. Start solving the LP!")
    lp_status = "unsafe"

    assert lower_bounds[-1].size(0) == 1, "only bounds with batch size 1"
    guro_start = time.time()
    # Assert that this is as expected a network with a single output
    orig_out_vars = m.net.final_node().solver_vars
    # assert len(orig_out_vars) == 1, "Network doesn't have scalar output"
    glbs = lower_bounds[-1].squeeze(0).clone()
    assert len(orig_out_vars) == len(rhs), "out shape not matching!"
    for out_idx in range(len(orig_out_vars)):
        # objVar = m.new_model.getVarByName(orig_out_vars[0].VarName)
        objVar = m.new_model.getVarByName(orig_out_vars[out_idx].VarName)
        decision_threshold = rhs[out_idx]

        m.new_model.setObjective(objVar, grb.GRB.MINIMIZE)
        m.new_model.update()
        m.new_model.optimize()

        # assert m.model_cut.status == 2, (
        #     f"model not optimized with status {m.model_cut.status}"
        # )
        if m.new_model.status == 2:
            glb = objVar.X
        elif m.new_model.status == 3:
            print("Model infeasible!")
            glb = float("inf")
        else:
            print(f"Warning: model status {m.new_model.status}!")
            glb = float("inf")
        print(
            f"glb [{objVar.VarName}, status={m.new_model.status}]:"
            + f"{glbs[0]} -> {glb}, against rhs {decision_threshold.item()}"
        )

        guro_end = time.time()
        print("Gurobi solved the LP with time", guro_end - guro_start)
        glbs[out_idx] = glb
        if glb > decision_threshold:
            lp_status = "safe"
            break

    # -- collect variables.
    vars = None

    # -- duals
    if m.new_model.status == 2:
        duals = {
            relu.name: torch.zeros(v.size(-1), 3)
            for relu, v in zip(m.net.relus, mask.values())
        }
        for constr in m.new_model.getConstrs():
            if constr.ConstrName.startswith("ReLU"):
                constr_name = constr.ConstrName.split("_")
                layer_name = constr_name[0][4:]
                constr_idx = int(constr_name[-1])
                node_idx = int(constr_name[1])

                duals[layer_name][node_idx][constr_idx] = constr.getAttr("Pi")
            else:
                pass  # other dual vars?

        # -- primals

        new_model_vars = {v.VarName: v for v in m.new_model.getVars()}

        layers = [m.net.final_node()]
        node = m.net.final_node()
        while node.inputs:
            layers = [node.inputs[0]] + layers
            node = node.inputs[0]

        primals = []
        for i, layer in enumerate(layers):
            pv = []
            layer_vars = layer.solver_vars
            if not isinstance(layer_vars[0], list):
                for j in range(len(layer_vars)):
                    pv.append(new_model_vars[layer_vars[j].VarName].X)
            else:
                for chan in range(len(layer_vars)):
                    for row in range(len(layer_vars[chan])):
                        for col in range(len(layer_vars[chan][row])):
                            pv.append(
                                new_model_vars[layer_vars[chan][row][col].VarName].X
                            )
            primals.append(torch.tensor(pv))

        vars = LPVars(primals[0], primals[1:], [d for d in duals.values()])

    del m.new_model
    print("LP status:", lp_status)
    return lp_status, glbs, vars
