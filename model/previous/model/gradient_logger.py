import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import dill
import fnmatch
from torch.utils.tensorboard import SummaryWriter

class GradientLogger:
    """
    GradientLogger class to log and visualize gradients of specified parameters in a PyTorch model.
    """
    def __init__(self, file, save_interval=100, window_size=10, 
                 base_width=6, base_height=5, width_scale=0.3, height_scale=0.2, 
                 filter_params=[],
                 exit_step=np.inf):
        """
        Args:
            file (str): ファイル保存パス
            save_interval (int): 何ステップごとに保存するか
            window_size (int): 平均を取る範囲
            base_width (int): グラフの基本幅
            base_height (int): グラフの基本高さ
            width_scale (float): パラメータ数に応じた幅の増加率
            height_scale (float): 最大ラベル長に応じた高さの増加率
        """
        self.gradients = {}  # Dictionary to store gradients for specified parameters
        self.steps = []      # List to store global steps
        self.epochs = []     # List to store epochs corresponding to steps
        self.save_interval = save_interval
        self.window_size = window_size
        self.base_width = base_width
        self.base_height = base_height
        self.height = base_height
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.filter_params = filter_params
        self.exit_step = exit_step
        self.file = file

    def __len__(self):
        return len(self.steps)
        

    def log(self, model, step, epoch, writer: SummaryWriter):
        """
        Log gradients of specified parameters in the model.

        Args:
            model (torch.nn.Module): The PyTorch model.
        """
        self._log_step(step, epoch)

        for name, param in model.named_parameters():
            # Skip parameters that match the filter patterns
            if any(fnmatch.fnmatch(name, pattern) for pattern in self.filter_params):
                continue  

            # Skip parameters that do not require gradients
            if not param.requires_grad:
                continue  

            # Initialize gradient tracking if the parameter is not yet in self.gradients
            if name not in self.gradients:
                self.gradients[name] = [0.0] * (len(self.steps) - 1)  # Fill missing values
            
            # Compute gradient norm if available, otherwise set to 0
            grad_norm = param.grad.norm().item() if param.grad is not None else 0.0

            # Store gradient norm
            self.gradients[name].append(grad_norm)
            
            # Log to tensorboard
            writer.add_scalar(f"{name}", grad_norm, step)


        if (len(self)-1) % self.save_interval == 0:
            if len(self) >= self.window_size:
                window_size = self.window_size
            else:
                window_size = len(self)
            avg_gradients = {
                name: np.mean(self.gradients[name][-window_size:])  # 直近 window_size ステップの平均
                for name in self.gradients
            }
            

            num_params = len(avg_gradients)  # パラメータ数
            max_label_length = max(len(name) for name in avg_gradients.keys())  # 最大ラベル長

            dynamic_width = self.base_width + num_params * self.width_scale  # パラメータ数に応じた幅
            dynamic_height = self.base_height + max_label_length * self.height_scale  # ラベル長に応じた高さ

            # Matplotlib で棒グラフを作成（サイズ調整）
            fig, ax = plt.subplots(figsize=(dynamic_width, dynamic_height))
            ax.bar(avg_gradients.keys(), avg_gradients.values(), color='blue')
            ax.set_yscale('log')  
            ax.set_ylabel("Gradient Norm (Average, Log Scale)")
            ax.set_title(f"Gradient Average (Last {self.window_size} steps)")

            # **Y軸のラベルを 10^n ごとに表示**
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=100))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$"))

            # **Y軸に 10^0.5 ごとのメモリ線を追加**
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.array([np.sqrt(10)]), numticks=100))
            ax.yaxis.grid(True, linestyle="-", alpha=0.7)
            ax.yaxis.grid(True, which='minor', linestyle="--", alpha=0.7)  

            # X軸ラベルを縦表示
            labels = list(avg_gradients.keys())
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90)

            plt.tight_layout()  # レイアウト調整（文字が見切れないように）

            # TensorBoard に棒グラフを記録
            writer.add_figure("Gradient_Averages", fig, step)
            plt.close(fig) 

            if step >= self.exit_step:
                exit()


    def _log_step(self, step, epoch):
        """
        Log the current global step and corresponding epoch.

        Args:
            step (int): The global step value.
            epoch (float): The epoch value (can be fractional).
        """
        self.steps.append(step)
        self.epochs.append(epoch)


    def plot_gradients(self, filter_params=[], xlim=None, ylim=None):
        """
        Plot gradients for each parameter individually over global steps.
        """
        if not self.gradients:
            print("No gradients recorded. Please ensure gradients are being tracked.")
            return

        for name, values in self.gradients.items():
            if name in filter_params:
                continue
            plt.figure(figsize=(10, 6))

            # Plot gradients for the current parameter
            plt.plot(self.steps, values, label=f"Gradient of {name}")
            
            # Configure the plot
            plt.xlabel("Global Step")
            plt.ylabel("Gradient Norm")
            plt.title(f"Gradient Norm vs. Global Steps for {name}")
            plt.legend()
            plt.grid(True)

            if xlim:
                plt.xlim(xlim)
            
            if ylim:
                plt.ylim(ylim)
            
            # Show the plot
            plt.show()

    def save(self, file_path):
        """
        Save the current state of the GradientLogger to a file.

        Args:
            file_path (str): Path to the file where the logger will be saved.
        """
        with open(file_path, "wb") as file:
            dill.dump(self, file)

    @staticmethod
    def load(file_path) -> 'GradientLogger':
        """
        Load a GradientLogger from a file.

        Args:
            file_path (str): Path to the file where the logger is stored.

        Returns:
            GradientLogger: The loaded GradientLogger instance.
        """
        with open(file_path, "rb") as file:
            return dill.load(file)