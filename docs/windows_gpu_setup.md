# Windows GPU Setup

Bu dokuman, bu makinede uygulanan genel CUDA ve GPU destekli PyTorch kurulumunu tekrar etmek icin hazirlandi. Kurulum proje-ozel degildir; ayni ortam baska derin ogrenme projelerinde de kullanilabilir.

## Mevcut Durum
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`
- Driver dogrulama: `nvidia-smi`
- Kurulu CUDA Toolkit: `13.2`
- Kurulu reusable ortam: `%USERPROFILE%\dl-gpu-py313`
- Jupyter kernel adi: `dl-gpu-py313`

Not: Ilk planda `CUDA Toolkit 13.1` hedeflenmisti. Gercek uygulamada resmi `winget` paketi `13.2` sundugu icin `13.2` kuruldu. Bu degisiklik surucu ve PyTorch GPU wheel'i ile uyumludur.

## 1. Surucu Dogrulama
PowerShell:

```powershell
nvidia-smi
```

Beklenen sonuc: GPU adi gorunmeli ve `Driver Version` alani dolu olmali.

## 2. CUDA Toolkit Kurulumu
PowerShell:

```powershell
winget install --id Nvidia.CUDA -e --accept-package-agreements --accept-source-agreements --disable-interactivity
```

Bu komut bu makinede `CUDA Toolkit 13.2` kurdu. NVIDIA belgelerine gore Windows display driver, CUDA 13.1 ve sonrasinda toolkit paketinin icinde gelmedigi icin toolkit kurulumu ile surucu kurulumunu ayri dusunmek gerekir.

Kurulum sonrasi yeni bir PowerShell penceresinde:

```powershell
nvcc --version
where.exe nvcc
[Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
```

Onemli: Toolkit kurulumu `PATH` degiskenini gunceller. Bu nedenle mevcut acik terminal oturumlari ve mevcut Codex oturumu eski `PATH` ile devam edebilir; `nvcc` komutunu yalnizca yeni acilan bir terminalde dogrulayin.

Bu makinede beklenen `nvcc` yolu:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe
```

## 3. Reusable Python Ortami
Python 3.13 bu makinede zaten kurulu oldugu icin reusable ortam su sekilde olusturuldu:

```powershell
python -m venv "$env:USERPROFILE\dl-gpu-py313"
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" -m pip install --upgrade pip
```

## 4. Genel Bilimsel Paketler
Asagidaki paketler genel kullanim icin yuklendi:

```powershell
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" -m pip install `
  jupyterlab ipykernel numpy pandas matplotlib scikit-learn seaborn
```

## 5. GPU Destekli PyTorch
PyTorch tarafinda resmi Windows `pip` wheel kurulumu kullanildi. Mevcut `Get Started` akisi Windows icin `cu128` secenegini sundugu icin GPU build su komutla yuklendi:

```powershell
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" -m pip install `
  torch torchvision torchaudio `
  --index-url https://download.pytorch.org/whl/cu128 `
  --extra-index-url https://pypi.org/simple
```

Bu makinede kurulu sonuc:
- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torchaudio 2.11.0+cu128`

## 6. Jupyter Kernel Kaydi
```powershell
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" -m ipykernel install --user `
  --name dl-gpu-py313 `
  --display-name "Python (dl-gpu-py313)"
```

## 7. Dogrulama
Toolkit dogrulamasi:

```powershell
nvcc --version
nvidia-smi
```

PyTorch dogrulamasi:

```powershell
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" -c "import torch; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Bu makinede alinan dogrulama:

```text
torch 2.11.0+cu128
torch_cuda 12.8
cuda_available True
device_name NVIDIA GeForce RTX 4070 Laptop GPU
```

Kisa GPU smoke testi:

```powershell
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" -c "import torch; a=torch.randn((1024,1024), device='cuda'); b=torch.randn((1024,1024), device='cuda'); print((a@b).mean().item())"
```

## 8. Project2 Kullanimi
`project2` icinde calismak icin:

```powershell
cd "D:\Ankara Universitesi\Ankara Uni 2025-2026\Bahar\Derin Ogrenme\deeplearning_lab\project2"
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" run_experiments.py
```

## Resmi Kaynaklar
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/)
- [NVIDIA CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)
