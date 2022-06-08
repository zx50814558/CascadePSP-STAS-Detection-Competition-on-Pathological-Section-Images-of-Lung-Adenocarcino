# CascadePSP-STAS-Detection-Competition-on-Pathological-Section-Images-of-Lung-Adenocarcinoma-II
STAS Detection Competition on Pathological Section Images of Lung Adenocarcinoma II: Using Image Segmentation to Cut STAS Contours
# 環境
```
python == 3.6.13
pytorch == 1.10.2
opencv-python == 4.5.5.64
progressbar2 == 3.55.0
tensorboard == 2.9.0
pandas == 1.1.5
```
# 安裝指令
```
conda create --name cascade python=3.6
conda activate cascade
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install progressbar2
pip install opencv-python
pip install tensorboard
pip install pandas
```

# Testing
Pretrained Models : https://drive.google.com/file/d/15SGC0mMRBiofohXov_Bq5RJfL7whgh6S/view?usp=sharing
Testing dataset : https://drive.google.com/file/d/1-CfW7BBceDxw2gkui-LYwrWJoXMw9Y5W/view?usp=sharing
Pretrained Models 下載後放至專案的根目錄  
Testing dataset 下載後放至專案的根目錄

Qucik Start:
```
1. 下載 Pretrained Models
2. 下載 Testing dataset
3. 執行下方指令
```

自己建立 Testing dataset :
```
1. 下載 Pretrained Models
2. 建立名稱為 input 的資料夾
3. 將 Semask 輸出圖片以及競賽的 Public 與 private dataset 放入 input 資料夾
4. 執行 convert.py
5. 執行下方指令
```

------------
```
python eval.py --dir input --model model_50000 --output output
```
# Training
Download the dataset:

900 張: https://drive.google.com/file/d/1e9fCU-H2HU1mL4IFLlrcCAWbp02_4eYK/view?usp=sharing

1053 張: https://drive.google.com/file/d/1d8PBd0uYv3KwKNzUcOCn1UGv0T_ygnHV/view?usp=sharing  
下載後須至```./train.py ```第 47 行更改為對應的資料路徑

------------
### 訓練方法:
- 這部分使用 900 張的資料集
1. 下載官方的 Pretrained Models: https://drive.google.com/file/d/1FMmUYtWsZB4fReoQmtqqn-NOZrC8CfWK/view
2. 至 ```./util/hyper_para.py ``` 裡更改  ```--load``` 路徑為下載的 ```Pretrained Models``` 路徑
3. 至 ```./util/hyper_para.py ``` 裡更改  ```--lr  ``` 為 2.25e-4
4. 執行  ```python train.py testing ``` 取第 34950 次的權重
- 這部分使用 1053 張的資料集
1. 至 ```./util/hyper_para.py ``` 裡更改  ```--load``` 路徑為 ```weights/testing/model_34950 ```
2. 至 ```./util/hyper_para.py ``` 裡更改  ```--lr  ``` 為 1.125e-4
3. 執行  ```python train.py testing_2 ``` 取第 50000 次的權重

------------

