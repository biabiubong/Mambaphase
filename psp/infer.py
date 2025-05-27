import torch
from torch.nn.utils.rnn import pad_sequence
from sequence_embedding import SequenceToVectorModel
import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self, sequence_model, mlp_units=512, dropout_rate=0.4):
        super().__init__()
        self.sequence_model = sequence_model
        self.mlp1 = nn.Sequential(
            nn.Linear(2560, mlp_units),
            nn.BatchNorm1d(mlp_units),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_units, mlp_units//2),
            nn.BatchNorm1d(mlp_units//2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_units//2, mlp_units//4),
            nn.BatchNorm1d(mlp_units//4),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.Linear(mlp_units//4 + 256, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids, rdict_seqs):
        embeddings = self.sequence_model(input_ids)
        rdict_emb = self.mlp1(rdict_seqs)
        combined = torch.cat([rdict_emb, embeddings], dim=1)
        return self.classifier(combined)

class Predictor:
    def __init__(self, model_path, device='cuda:1'):
        self.device = torch.device(device)
        self.best_hyperparams = {
            'd_model': 256,
            'd_inner': 128,
            'n_ssm': 8,
            'dt_rank': 8,
            'n_layer': 1,
            'dropout': 0.1,
            'mlp_units': 1024,
            'dropout_rate': 0.3
        }
        self.AA_LIST = "ACDEFGHIKLMNPQRSTVWYU"
        self.amino_acid_to_index = {aa: idx for idx, aa in enumerate(self.AA_LIST)}
        self.cls_model = self._load_model(model_path)
        self.cls_model.eval()

    def _load_model(self, model_path):
        sequence_model = SequenceToVectorModel(
            vocab_size=len(self.AA_LIST),
            d_model=self.best_hyperparams['d_model'],
            d_inner=self.best_hyperparams['d_inner'],
            n_ssm=self.best_hyperparams['n_ssm'],
            dt_rank=self.best_hyperparams['dt_rank'],
            n_layer=self.best_hyperparams['n_layer'],
            dropout=self.best_hyperparams['dropout'],
            output_dim=256
        )
        model = ClassificationModel(
            sequence_model,
            mlp_units=self.best_hyperparams['mlp_units'],
            dropout_rate=self.best_hyperparams['dropout_rate']
        ).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        return model

    def predict(self, esm_reps):
        input_ids = [torch.zeros(1, dtype=torch.long) for _ in esm_reps]
        padded_seqs = pad_sequence(input_ids, batch_first=True, padding_value=0).to(self.device)
        esm_tensor = torch.stack(esm_reps).to(self.device)
        
        with torch.no_grad():
            outputs = self.cls_model(padded_seqs, esm_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

# # 使用示例
# if __name__ == "__main__":
#     # 初始化预测器
#     predictor = Predictor(cls_model_path, device='cuda:1')
    
#     # 示例输入（假设esm_rep是预计算好的）
#     esm_reps = [torch.randn(2560) for _ in range(5)]  # 替换为实际的esm_rep列表
    
#     # 获取概率
#     probabilities = predictor.predict(esm_reps)
#     print(probabilities)