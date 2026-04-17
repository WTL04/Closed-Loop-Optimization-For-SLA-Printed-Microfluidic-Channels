# CAD to CFD Pipeline (OpenFOAM + ParaView)

## Overview

This guide explains how to:

* Install WSL + Ubuntu
* Install OpenFOAM
* Run CFD simulation
* Visualize results in ParaView
* Extract flow rate

This assumes:

* You have cloned this repository
* The `cfd/channelCase` folder already exists with all required files

---

# 1. Install WSL + Ubuntu (Windows)

Open **PowerShell as Administrator**:

```powershell
wsl --install
```

Restart PC if prompted.

After restart:

* Open **Ubuntu**
* Create username + password

Check installation:

```powershell
wsl --list --verbose
```

---

# 2. Open Ubuntu and install dependencies

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y nano git wget curl unzip software-properties-common
```

---

# 3. Install OpenFOAM

```bash
sudo apt install -y openfoam
```

Verify installation:

```bash
which blockMesh
which snappyHexMesh
which simpleFoam
```

---

# 4. Fix OpenFOAM environment (IMPORTANT)

If commands fail, run:

```bash
export WM_PROJECT=OpenFOAM
export WM_PROJECT_DIR=/usr/share/openfoam
export FOAM_ETC=/usr/share/openfoam/etc
```

Make it permanent:

```bash
echo 'export WM_PROJECT=OpenFOAM' >> ~/.bashrc
echo 'export WM_PROJECT_DIR=/usr/share/openfoam' >> ~/.bashrc
echo 'export FOAM_ETC=/usr/share/openfoam/etc' >> ~/.bashrc
source ~/.bashrc
```

---

# 5. Install ParaView 

sudo apt install -y paraview

---

# 6. Navigate to your CFD case

Clone or open your repo, then:

In Ubuntu:

cd <your-repo-name>

Your repo structure should look like:

<repo-root>/
├── ax/
│   └── src/
├── cfd/
│   └── channelCase/

---
# 7. Ensure STL is present

```bash
mkdir -p constant/triSurface
ls constant/triSurface
```

You should see:

```text
channels_fluid.stl
```

---

# 8. Build mesh

```bash
rm -rf constant/polyMesh
blockMesh
surfaceFeatureExtract
snappyHexMesh -overwrite
checkMesh
```

---

# 9. Run simulation

```bash
simpleFoam
```

---

# 10. Create ParaView file

```bash
touch case.foam
paraview
```

---

# 11. In ParaView

1. File → Open → select `case.foam`
3. Click **Apply**

---

# 12. Visualize flow

In ParaView:

* Set:

```text
Coloring → U → Magnitude
Representation → Surface
```

---

# 13. Extract flow rate (IMPORTANT)

## Step 1 — Create slice

```text
Filters → Slice → Apply
```

## Step 2 — Set slice

```text
Normal = (1, 0, 0)
Origin = (0.004, 0.0001, 0.0006)
```

Adjust ONLY `Origin X` slightly if needed.

---

## Step 3 — Integrate

```text
Filters → Integrate Variables → Apply
```

---

## Step 4 — Read result

```text
View → Spreadsheet View
```

Look at:

```text
U = (Ux, Uy, Uz)
```

👉 Flow rate = **Ux**

---

# 14. IMPORTANT RULES

* Always run commands from:

```bash
cfd/channelCase
```

* After re-running simulation:

```text
DELETE old Slice + Integrate
CREATE new ones
```
---

# 15. Full command sequence (quick run)

```bash
cd /mnt/c/.../cfd/channelCase

rm -rf constant/polyMesh
blockMesh
surfaceFeatureExtract
snappyHexMesh -overwrite
checkMesh
simpleFoam
touch case.foam
paraview
```

Then open `case.foam` in ParaView.

---

# 16. Final workflow

```text
Run CFD → Open ParaView → Slice → Integrate → Read Ux
```

---

# Done

You now have:

* Working CFD simulation
* Correct slicing
* Flow rate extraction

---

If anything breaks, rerun:

```bash
rm -rf constant/polyMesh
blockMesh
snappyHexMesh -overwrite
simpleFoam
```

---
