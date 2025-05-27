import torch
import numpy as np
from torch import nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=2560, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class PHpredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPClassifier().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def _preprocess(self, features):
        """统一输入格式为二维张量"""
        if not isinstance(features, (list, np.ndarray, torch.Tensor)):
            features = [features]
            
        # 转换所有输入为张量
        processed = []
        for feat in features:
            if isinstance(feat, np.ndarray):
                tensor = torch.FloatTensor(feat)
            elif isinstance(feat, torch.Tensor):
                tensor = feat.float()
            else:
                raise ValueError("支持数据类型: numpy数组/PyTorch张量")
            
            # 维度处理 (N, 2560)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() > 2:
                raise ValueError(f"非法维度: {tensor.shape}，期望形状应为 (batch_size, 2560)")
                
            processed.append(tensor)
            
        return torch.cat(processed).to(self.device)
    
    def predict(self, esm_reps):
        """返回形状为 (batch_size, 3) 的概率矩阵"""
        inputs = self._preprocess(esm_reps)
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

