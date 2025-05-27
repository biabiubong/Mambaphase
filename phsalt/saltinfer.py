import torch
import numpy as np
import pandas as pd
from torch import nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
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

class saltPredictor:
    def __init__(self, model_path):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型初始化
        self.model = MLPClassifier(2560, 3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _preprocess_features(self, features):
        """统一输入特征格式为 (batch_size, 2560) 的tensor"""
        # 转换单样本输入为列表
        if not isinstance(features, (list, np.ndarray, torch.Tensor)):
            features = [features]
        
        processed = []
        for feat in features:
            # 类型转换
            if isinstance(feat, np.ndarray):
                tensor = torch.FloatTensor(feat)
            elif isinstance(feat, torch.Tensor):
                tensor = feat.float()
            else:
                raise ValueError("仅支持numpy数组/PyTorch张量")
            
            # 维度处理
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)  # (2560) -> (1, 2560)
            elif tensor.dim() != 2:
                raise ValueError(f"非法输入维度: {tensor.shape}")
                
            processed.append(tensor)
            
        return torch.cat(processed).to(self.device)

    def predict(self, esm_reps):
        """
        执行预测并返回结果DataFrame
        返回值包含两列:
        - prediction: 预测类别 (0, 1, 2)
        - probabilities: 各类别概率数组
        """
        # 数据预处理
        inputs = self._preprocess_features(esm_reps)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
        
        # 转换为numpy并封装结果
        return pd.DataFrame({
            "prediction": preds.cpu().numpy(),
            "probabilities": probs.cpu().numpy().tolist()
        })