# SuperVAR


## Setup Instructions

### 1. Update system packages

```bash
sudo apt update
```

---

### 2. Install Python 3.10 virtual environment support

If you don't have it already, install `python3.10-venv`:

```bash
sudo apt install python3.10-venv -y
```

---

### 3. Create a Python 3.10 virtual environment

Create the virtual environment in your home directory under `~/envs/py310`:

```bash
python3.10 -m venv ~/envs/py310
```

---

###  4. Activate the virtual environment

```bash
source ~/envs/py310/bin/activate
```
---

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Download Dataset and Model

```bash
dataprepare.ipynb
```