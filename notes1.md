### Save figures
 ```
from pathlib import Path

images_path = Path('images','mnist')
if not images_path.is_dir():
    images_path.mkdir(parents=True, exist_ok=True)

def save_fig(fig_name, tight_layout=True, fig_extension='png', resolution=300):
    image_path = Path(images_path,f'{fig_name}.{fig_extension}')
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path,format=fig_extension, dpi=resolution)
```

### Download Data (tgz)

```
import os

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
  if not os.path.isdir(housing_path):
    os.makedirs(housing_path)
  tgz_path = os.path.join(housing_path, "housing.tgz")
  urllib.request.urlretrieve(housing_url, tgz_path)
  housing_tgz = tarfile.open(tgz_path)
  housing_tgz.extractall(path=housing_path)
  housing_tgz.close()
```
### Pyplot Configuration 
```
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```
### Sublots
```
plt.figure(figsize=(9, 9))
for idx, image_data in enumerate(X[:100]):
    plt.subplot(10, 10, idx + 1)
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
```
or
```
fig,axes = plt.subplots(nrows=10,ncols=10,figsize=(9,9))
for idx, ax in enumerate(axes.flat):
    ax.imshow(X[idx].reshape(28,28),cmap='binary')
    ax.axis('off')
plt.subplots_adjust(wspace=0,hspace=0)
plt.show()
```
### Execution Time Measurement
```
%timeit gaussian_rnd_proj.transform(X)
```

