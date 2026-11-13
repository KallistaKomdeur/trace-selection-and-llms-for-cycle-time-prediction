import os
import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('TwoLayerLinearEdge')
class TwoLayerLinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()       
        # The few-shot campaign rebuilds the edge features per test case, so their
        # width changes from run to run and cannot be hardcoded per dataset name.
        _ovr = os.environ.get('PGTNET_EDGE_IN_DIM')
        if _ovr:
            self.in_dim = int(_ovr)
        elif cfg.dataset.name in ['BPIC12CWcycletimeprediction' , 'BPIC12Ocycletimeprediction',
                                  'BPIC12Wcycletimeprediction']:
            self.in_dim = 68 
        elif cfg.dataset.name in ['BPIC12cycletimeprediction' , 'BPIC12Ccycletimeprediction']:
            self.in_dim = 77
        elif cfg.dataset.name == 'BPIC12Acycletimeprediction':
            self.in_dim = 69
        elif cfg.dataset.name == 'BPIC13Ccycletimeprediction':
            self.in_dim = 93          
        elif cfg.dataset.name == 'BPIC13Icycletimeprediction':
            self.in_dim = 115
        elif cfg.dataset.name == 'BPIC20Dcycletimeprediction':
            self.in_dim = 17
        elif cfg.dataset.name == 'BPIC20Icycletimeprediction':
            self.in_dim = 407
        elif cfg.dataset.name == 'ENVPERMITcycletimeprediction':
            self.in_dim = 120
        elif cfg.dataset.name == 'HELPDESKcycletimeprediction':
            self.in_dim = 477
        elif cfg.dataset.name == 'HOSPITALcycletimeprediction':
            self.in_dim = 150
        elif cfg.dataset.name == 'SEPSIScycletimeprediction':
            self.in_dim = 250
        elif cfg.dataset.name == 'Trafficfinescycletimeprediction':
            self.in_dim = 267
        elif cfg.dataset.name == 'BPIC15M1cycletimeprediction':
            self.in_dim = 219 
        elif cfg.dataset.name == 'BPIC15M2cycletimeprediction':
            self.in_dim = 142 
        elif cfg.dataset.name == 'BPIC15M3cycletimeprediction':
            self.in_dim = 209 
        elif cfg.dataset.name == 'BPIC15M4cycletimeprediction':
            self.in_dim = 141
        elif cfg.dataset.name == 'BPIC15M5cycletimeprediction':
            self.in_dim = 208
        elif cfg.dataset.name == 'HELPDESKV2cycletimeprediction':
            self.in_dim = 36
        elif cfg.dataset.name == 'CONFIDENTIALcycletimeprediction':
            self.in_dim = 18
        elif cfg.dataset.name == 'PRODUCTIONcycletimeprediction':
            self.in_dim = 107
        elif cfg.dataset.name == 'BPI11HALFcycletimeprediction':
            self.in_dim = 53
        elif cfg.dataset.name == 'PURCHASINGcycletimeprediction':
            self.in_dim = 41
        elif cfg.dataset.name == 'BPI12Wcycletimeprediction':
            self.in_dim = 73
        elif cfg.dataset.name == 'BASELINEcycletimeprediction':
            self.in_dim = 68
        elif cfg.dataset.name == 'BPIC2011cycletimeprediction':
            self.in_dim = 1
        elif cfg.dataset.name == 'BPIC2015cycletimeprediction':
            self.in_dim = 1
        elif cfg.dataset.name == 'BASELINEDRIFTEDcycletimeprediction':
            self.in_dim = 68
        elif cfg.dataset.name == 'BPIC20PREPAIDcycletimeprediction':
            self.in_dim = 17
        elif cfg.dataset.name == 'BPIC20INTERNATIONALV2cycletimeprediction':
            self.in_dim = 17
        elif cfg.dataset.name == 'BPIC20RFPcycletimeprediction':
            self.in_dim = 17
        elif cfg.dataset.name == 'BPIC20PERMITcycletimeprediction':
            self.in_dim = 18
        elif cfg.dataset.name == 'BPIC20DOMESTICV2cycletimeprediction':
            self.in_dim = 17
        elif cfg.dataset.name == 'BPIC12V2cycletimeprediction':
            self.in_dim = 83
        elif cfg.dataset.name == 'HOSPITALBILLINGcycletimeprediction':
            self.in_dim = 667
        elif cfg.dataset.name == 'TRAFFICFINESV2cycletimeprediction':
            self.in_dim = 168
        elif cfg.dataset.name == 'BPIC17cycletimeprediction':
            self.in_dim = 178
        elif cfg.dataset.name == 'BPIC20IPAPERcycletimeprediction':
            self.in_dim = 407
        elif cfg.dataset.name == 'BPIC20RFPPAPERcycletimeprediction':
            self.in_dim = 736
        elif cfg.dataset.name == 'BPIC20PERMITPAPERcycletimeprediction':
            self.in_dim = 51
        # extra condition for ablation study:
        elif 'ablation' in cfg.dataset.name:
            self.in_dim = 6
        else:
            raise ValueError("Input edge feature dim is required to be hardset "
                             "or refactored to use a cfg option.")
            
        self.encoder1 = torch.nn.Linear(self.in_dim, int((emb_dim+self.in_dim)/2))
        self.encoder2 = torch.nn.Linear(int((emb_dim+self.in_dim)/2), emb_dim)

    def forward(self, batch):
        batch.edge_attr = self.encoder1(batch.edge_attr.view(-1, self.in_dim))
        batch.edge_attr = self.encoder2(batch.edge_attr)
        return batch