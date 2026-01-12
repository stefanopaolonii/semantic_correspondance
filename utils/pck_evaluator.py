import torch
import numpy as np

from .geometry import *


class PCKEvaluator:

    def __init__(self, cfg) -> None:
        
        self.alpha = cfg.EVALUATOR.ALPHA
        self.by = cfg.EVALUATOR.BY
        self.method_options = ('nn', 'bilinear', 'softmax', 'kernelsoftmax')

        self.result = {}
        for method_name in self.method_options:
            for alpha in self.alpha:
                self.result.update({f'{method_name}_pck{alpha}': {"all": []}})

    def clear_result(self):
        self.result = {key : {'all': []} for key in self.result}


    def calculate_pck(self, trg_kps, matches, n_pts, categories, pckthres, method_name):
        '''
        trg_kps [torch.Tensor] (BxNx2)
        matches [torch.Tensor] (BxNx2)
        n_pts[torch.Tensor] (B)
        pckthres[float] (B)
        '''
        B = trg_kps.shape[0]
        
        for b in range(B):
            npt = n_pts[b].item()
            thres = pckthres[b].item()
            category = categories[b]

            tkps = trg_kps[b, :npt]     # npt x 2
            mats = matches[b, :npt]     # npt x 2

            diff = torch.norm(tkps-mats, dim=-1)
            
            for alpha in self.alpha:
                if self.by == 'image':
                    pck = (diff <= alpha*thres).float().mean()
                    if category in self.result[f'{method_name}_pck{alpha}']:
                        self.result[f'{method_name}_pck{alpha}'][category].append(pck.item())
                    else:
                        self.result[f'{method_name}_pck{alpha}'][category] = []
                        self.result[f'{method_name}_pck{alpha}'][category].append(pck.item())
                    self.result[f'{method_name}_pck{alpha}']["all"].append(pck.item())
                elif self.by == "point":
                    pck = (diff <= alpha*thres).float().tolist()
                    if category in self.result[f'{method_name}_pck{alpha}']:
                        self.result[f'{method_name}_pck{alpha}'][category].extend(pck)
                    else:
                        self.result[f'{method_name}_pck{alpha}'][category] = []
                        self.result[f'{method_name}_pck{alpha}'][category].extend(pck)
                    self.result[f'{method_name}_pck{alpha}']["all"].extend(pck)
                else:
                    raise ValueError(f"select between ('image', 'point')")

    def summerize_result(self):
        out = {}
        for method_name in self.method_options:
            for alpha in self.alpha:
                out[f'{method_name}_pck{alpha}'] = {}
                for k, v in self.result[f'{method_name}_pck{alpha}'].items():
                    out[f'{method_name}_pck{alpha}'][k] = np.array(v).mean()
        return out
    
    def print_summarize_result(self):
        result = self.summerize_result()
        print(" " * 16 + "".join([f"{alpha:<10}" for alpha in self.alpha]))  # header
        for method_name in self.method_options:
            pcks = [f"{result[f'{method_name}_pck{alpha}']['all']:.2%}" for alpha in self.alpha]
            row = f"{method_name:<15}" + "".join([f"{pck:<10}" for pck in pcks])
            print(row)  # rows

    def save_result(self, save_file):
        result = self.summerize_result()
        outstring = ""
        for method_name in self.method_options:
            outstring += f"{method_name}:\n"
            catstring = ""
            for alpha in self.alpha:
                cat_list = []
                pck_list = []
                for k, v in result[f'{method_name}_pck{alpha}'].items():
                    if k != "all":
                        cat_list.append(k)
                        pck_list.append(v)
                cat_list = np.array(cat_list)
                pck_list = np.array(pck_list)
                indices = np.argsort(cat_list)
                cat_list = cat_list[indices]
                pck_list = pck_list[indices]
                cat_list = cat_list.tolist()
                pck_list = pck_list.tolist()
                pck_list = [f"{pck:.2%}" for pck in pck_list]
                cat_list.append("all")
                pck_list.append(f"{result[f'{method_name}_pck{alpha}']['all']:.2%}")

                if len(catstring) == 0:
                    catstring += " " * 12 + "".join([f"{category:<12}" for category in cat_list]) + "\n"
                    outstring += catstring
                row = f"{alpha:<12}" + "".join([f"{pck:<12}" for pck in pck_list]) + "\n"
                outstring += row

            outstring += "-----------------------------------------------------------------\n"

        with open(save_file, "w") as f:
            f.write(outstring)
